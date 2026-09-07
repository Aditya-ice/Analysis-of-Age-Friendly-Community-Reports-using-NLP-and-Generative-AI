output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.app.name
}

output "database_connection_name" {
  value = google_sql_database_instance.postgres.connection_name
}

output "database_url_secret" {
  value = google_secret_manager_secret.database_url.secret_id
}

output "github_workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "github_deployer_service_account" {
  value = google_service_account.github_deployer.email
}
