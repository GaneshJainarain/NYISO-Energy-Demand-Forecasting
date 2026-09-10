output "api_url" {
  description = "Base URL of the deployed API."
  value = var.use_localstack ? (
    "${var.localstack_endpoint}/restapis/${aws_api_gateway_rest_api.api.id}/${aws_api_gateway_stage.api.stage_name}/_user_request_"
  ) : aws_api_gateway_stage.api.invoke_url
}

output "bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "lambda_name" {
  value = aws_lambda_function.predict.function_name
}
