variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "cluster_name" {
  type    = string
  default = "netanomaly-cluster"
}

variable "vpc_id" {
  description = "Existing VPC to deploy into — fill in for your account."
  type        = string
}

variable "subnet_ids" {
  description = "Subnets for the EKS node group — fill in for your account."
  type        = list(string)
}
