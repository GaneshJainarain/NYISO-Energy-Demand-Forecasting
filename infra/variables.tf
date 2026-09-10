variable "use_localstack" {
  description = "Point every AWS call at LocalStack instead of real AWS. Set false to deploy for real."
  type        = bool
  default     = true
}

variable "localstack_endpoint" {
  type    = string
  default = "http://localhost:4566"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "nyiso-peak-forecast"
}

variable "lambda_zip" {
  description = "Path to the built deployment package (see serving/build.sh)."
  type        = string
  default     = "../serving/lambda.zip"
}

variable "artifacts_dir" {
  description = "Local directory holding model.ubj / recent.json / metadata.json."
  type        = string
  default     = "../artifacts"
}

variable "lambda_architecture" {
  type    = string
  default = "arm64"
}
