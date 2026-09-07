variable "project_id" {
  description = "Google Cloud project ID."
  type        = string
}

variable "region" {
  description = "Region for Cloud Run, Cloud SQL, Storage, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "staging"
}

variable "container_image" {
  description = "Immutable Artifact Registry image reference for the API and jobs."
  type        = string
}

variable "github_repository" {
  description = "GitHub repository in owner/name format allowed to deploy."
  type        = string
  default     = "Aditya-ice/Analysis-of-Age-Friendly-Community-Reports-using-NLP-and-Generative-AI"
}

variable "database_tier" {
  description = "Cloud SQL machine tier."
  type        = string
  default     = "db-custom-1-3840"
}

variable "database_user" {
  description = "Application database user created in Cloud SQL."
  type        = string
  default     = "elderhelp"
}

variable "database_password" {
  description = "Application database password. Pass with TF_VAR_database_password; never commit it."
  type        = string
  sensitive   = true
}

variable "generation_model" {
  description = "Vertex AI generation model recorded with ingestion and request telemetry."
  type        = string
  default     = "gemini-3.6-flash"
}

variable "embedding_model" {
  description = "Vertex AI embedding model used for the entire corpus."
  type        = string
  default     = "gemini-embedding-2"
}

variable "document_ai_processor" {
  description = "Optional full Document AI processor resource name used for low-quality PDF pages."
  type        = string
  default     = null
  nullable    = true
}
