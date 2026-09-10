# NYISO peak-demand forecasting API.
#
#   S3 (artifacts) -> Lambda (inference) -> API Gateway (public HTTPS)
#
# Defaults to LocalStack so the whole thing runs for free. `use_localstack=false`
# targets real AWS with no other change - that is the point of keeping the
# endpoints in one conditional block rather than hardcoding them.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region

  # LocalStack accepts any credentials and has no real account to validate.
  access_key                  = var.use_localstack ? "test" : null
  secret_key                  = var.use_localstack ? "test" : null
  skip_credentials_validation = var.use_localstack
  skip_metadata_api_check     = var.use_localstack
  skip_requesting_account_id  = var.use_localstack
  s3_use_path_style           = var.use_localstack

  dynamic "endpoints" {
    for_each = var.use_localstack ? [1] : []
    content {
      s3         = var.localstack_endpoint
      lambda     = var.localstack_endpoint
      iam        = var.localstack_endpoint
      sts        = var.localstack_endpoint
      apigateway = var.localstack_endpoint
      logs       = var.localstack_endpoint
      events     = var.localstack_endpoint
    }
  }
}

# --- artifact storage ----------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket        = "${var.project}-artifacts"
  force_destroy = true
}

# The three files scripts/export_model.py produces. etag means terraform apply
# re-uploads whenever a retrain changes them.
resource "aws_s3_object" "artifact" {
  for_each = toset(["model.ubj", "recent.json", "metadata.json"])

  bucket = aws_s3_bucket.artifacts.id
  key    = "artifacts/${each.value}"
  source = "${var.artifacts_dir}/${each.value}"
  etag   = filemd5("${var.artifacts_dir}/${each.value}")
}

# --- lambda --------------------------------------------------------------

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.project}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}

data "aws_iam_policy_document" "lambda" {
  # Read-only on the artifact bucket - inference never writes.
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/*"]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.project}-lambda-policy"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# The zip is ~52 MB, over Lambda's 50 MB direct-upload limit, so it is staged
# in S3 and referenced from there. (xgboost drags in scipy, which is 144 MB of
# the unzipped total.)
resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.artifacts.id
  key    = "lambda/lambda.zip"
  source = var.lambda_zip
  etag   = filemd5(var.lambda_zip)
}

resource "aws_lambda_function" "predict" {
  function_name    = "${var.project}-predict"
  role             = aws_iam_role.lambda.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.13"
  architectures    = [var.lambda_architecture]
  s3_bucket        = aws_s3_bucket.artifacts.id
  s3_key           = aws_s3_object.lambda_zip.key
  source_code_hash = filebase64sha256(var.lambda_zip)

  # Cold start pulls ~750 KB from S3 and loads a 300-tree booster.
  timeout     = 30
  memory_size = 1024

  environment {
    variables = {
      ARTIFACT_BUCKET = aws_s3_bucket.artifacts.id
      ARTIFACT_PREFIX = "artifacts"
      # Inside LocalStack's Lambda container, localhost is not the host.
      S3_ENDPOINT_URL = var.use_localstack ? "http://localstack:4566" : ""
    }
  }

  depends_on = [aws_s3_object.artifact, aws_s3_object.lambda_zip]
}

# --- api gateway ---------------------------------------------------------

resource "aws_api_gateway_rest_api" "api" {
  name        = "${var.project}-api"
  description = "Day-ahead NYISO peak demand forecast"
}

resource "aws_api_gateway_resource" "proxy" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.proxy.id
  http_method   = "ANY"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "proxy" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.proxy.id
  http_method             = aws_api_gateway_method.proxy.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.predict.invoke_arn
}

resource "aws_api_gateway_method" "root" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_rest_api.api.root_resource_id
  http_method   = "ANY"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "root" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_rest_api.api.root_resource_id
  http_method             = aws_api_gateway_method.root.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.predict.invoke_arn
}

resource "aws_api_gateway_deployment" "api" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  triggers = {
    redeploy = sha1(jsonencode([
      aws_api_gateway_resource.proxy.id,
      aws_api_gateway_method.proxy.id,
      aws_api_gateway_integration.proxy.id,
      aws_api_gateway_integration.root.id,
    ]))
  }
  lifecycle {
    create_before_destroy = true
  }
  depends_on = [aws_api_gateway_integration.proxy, aws_api_gateway_integration.root]
}

resource "aws_api_gateway_stage" "api" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  deployment_id = aws_api_gateway_deployment.api.id
  stage_name    = "prod"
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.predict.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}
