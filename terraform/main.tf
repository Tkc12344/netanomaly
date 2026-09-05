# Starting point only — fill in variables.tf and a remote backend before
# this touches real infrastructure. Images come from GHCR (see
# .github/workflows/ci-cd.yml), not from a second registry here.
#
# What this defines:
#   - a minimal EKS cluster to run the k8s/ manifests against
#
# What it deliberately does NOT do: create a container registry (CI
# already pushes ghcr.io), manage secrets, or wire up DNS/TLS.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Use a remote backend (S3 + DynamoDB lock table, or Terraform Cloud)
  # before this touches real infrastructure — local state is fine for
  # learning, not for anything you'd rely on.
  # backend "s3" {
  #   bucket = "your-tfstate-bucket"
  #   key    = "netanomaly/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.31"

  vpc_id     = var.vpc_id
  subnet_ids = var.subnet_ids

  eks_managed_node_groups = {
    default = {
      instance_types = ["t3.medium"]
      min_size       = 1
      max_size       = 3
      desired_size   = 2
    }
  }
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "image_registry" {
  value       = "ghcr.io"
  description = "CI publishes netanomaly-api here; point k8s images at ghcr.io/<org>/netanomaly/netanomaly-api."
}
