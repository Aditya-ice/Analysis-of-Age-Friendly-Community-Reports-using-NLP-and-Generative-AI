locals {
  name = "elderhelp-${var.environment}"
  services = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "discoveryengine.googleapis.com",
    "documentai.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "sts.googleapis.com",
  ])
}

resource "google_project_service" "required" {
  for_each           = local.services
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "app" {
  location      = var.region
  repository_id = "elderhelp"
  format        = "DOCKER"
  depends_on    = [google_project_service.required]
}

resource "google_storage_bucket" "reports" {
  name                        = "${var.project_id}-${local.name}-reports"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning { enabled = true }
}

resource "google_sql_database_instance" "postgres" {
  name                = local.name
  region              = var.region
  database_version    = "POSTGRES_16"
  deletion_protection = true
  settings {
    tier              = var.database_tier
    availability_type = "ZONAL"
    disk_autoresize   = true
    disk_type         = "PD_SSD"
    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }
    ip_configuration { ipv4_enabled = true }
  }
  depends_on = [google_project_service.required]
}

resource "google_sql_database" "app" {
  name     = "elderhelp"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app" {
  name     = var.database_user
  instance = google_sql_database_instance.postgres.name
  password = var.database_password
}

resource "google_secret_manager_secret" "database_url" {
  secret_id = "${local.name}-database-url"
  replication {
    auto {}
  }
  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "database_url" {
  secret = google_secret_manager_secret.database_url.id
  secret_data = format(
    "postgresql+asyncpg://%s:%s@/elderhelp?host=/cloudsql/%s",
    var.database_user,
    urlencode(var.database_password),
    google_sql_database_instance.postgres.connection_name,
  )
  depends_on = [google_sql_user.app]
}

resource "google_service_account" "api" {
  account_id   = "${local.name}-api"
  display_name = "ElderHelp ${var.environment} API"
}

resource "google_service_account" "ingestion" {
  account_id   = "${local.name}-ingest"
  display_name = "ElderHelp ${var.environment} ingestion"
}

resource "google_service_account" "github_deployer" {
  account_id   = "${local.name}-github"
  display_name = "ElderHelp ${var.environment} GitHub deployer"
}

resource "google_project_iam_member" "api_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_discovery" {
  project = var.project_id
  role    = "roles/discoveryengine.user"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "ingestion_vertex" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_document_ai" {
  project = var.project_id
  role    = "roles/documentai.apiUser"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_storage_bucket_iam_member" "ingestion_storage" {
  bucket = google_storage_bucket.reports.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "cloud_sql_api" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "cloud_sql_ingestion" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_secret_manager_secret_iam_member" "api_database_url" {
  secret_id = google_secret_manager_secret.database_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret_iam_member" "ingestion_database_url" {
  secret_id = google_secret_manager_secret.database_url.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_cloud_run_v2_service" "api" {
  name     = local.name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account                  = google_service_account.api.email
    timeout                          = "60s"
    max_instance_request_concurrency = 20
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = var.container_image
      resources { limits = { cpu = "1", memory = "1Gi" } }
      ports { container_port = 8080 }
      env {
        name  = "ELDERHELP_ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "ELDERHELP_GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "ELDERHELP_GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "ELDERHELP_GENERATION_MODEL"
        value = var.generation_model
      }
      env {
        name  = "ELDERHELP_EMBEDDING_MODEL"
        value = var.embedding_model
      }
      env {
        name = "ELDERHELP_DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      startup_probe {
        http_get { path = "/healthz" }
        initial_delay_seconds = 2
        period_seconds        = 5
        failure_threshold     = 12
      }
      liveness_probe {
        http_get { path = "/healthz" }
        period_seconds = 30
      }
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
    volumes {
      name = "cloudsql"
      cloud_sql_instance { instances = [google_sql_database_instance.postgres.connection_name] }
    }
  }
  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.api_database_url,
    google_project_iam_member.cloud_sql_api,
  ]
}

resource "google_cloud_run_v2_job" "ingestion" {
  name     = "${local.name}-ingestion"
  location = var.region
  template {
    template {
      service_account = google_service_account.ingestion.email
      timeout         = "3600s"
      containers {
        image   = var.container_image
        command = ["elderhelp"]
        args    = ["ingest-reports", "/app"]
        env {
          name  = "ELDERHELP_ENVIRONMENT"
          value = var.environment
        }
        env {
          name  = "ELDERHELP_GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        env {
          name  = "ELDERHELP_GOOGLE_CLOUD_LOCATION"
          value = var.region
        }
        env {
          name  = "ELDERHELP_STORAGE_BUCKET"
          value = google_storage_bucket.reports.name
        }
        env {
          name  = "ELDERHELP_GENERATION_MODEL"
          value = var.generation_model
        }
        env {
          name  = "ELDERHELP_EMBEDDING_MODEL"
          value = var.embedding_model
        }
        dynamic "env" {
          for_each = var.document_ai_processor == null ? [] : [var.document_ai_processor]
          content {
            name  = "ELDERHELP_DOCUMENT_AI_PROCESSOR"
            value = env.value
          }
        }
        env {
          name = "ELDERHELP_DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
      volumes {
        name = "cloudsql"
        cloud_sql_instance { instances = [google_sql_database_instance.postgres.connection_name] }
      }
    }
  }
  depends_on = [
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.ingestion_database_url,
    google_project_iam_member.cloud_sql_ingestion,
  ]
}

resource "google_cloud_run_v2_job" "migration" {
  name     = "${local.name}-migration"
  location = var.region
  template {
    template {
      service_account = google_service_account.ingestion.email
      timeout         = "900s"
      containers {
        image   = var.container_image
        command = ["alembic"]
        args    = ["upgrade", "head"]
        env {
          name  = "ELDERHELP_ENVIRONMENT"
          value = var.environment
        }
        env {
          name = "ELDERHELP_DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.database_url.secret_id
              version = "latest"
            }
          }
        }
        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
      volumes {
        name = "cloudsql"
        cloud_sql_instance { instances = [google_sql_database_instance.postgres.connection_name] }
      }
    }
  }
  depends_on = [
    google_secret_manager_secret_version.database_url,
    google_secret_manager_secret_iam_member.ingestion_database_url,
    google_project_iam_member.cloud_sql_ingestion,
  ]
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "${local.name}-github"
  display_name              = "ElderHelp GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository == '${var.github_repository}'"
  oidc { issuer_uri = "https://token.actions.githubusercontent.com" }
}

resource "google_service_account_iam_member" "github_identity" {
  service_account_id = google_service_account.github_deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

resource "google_project_iam_member" "github_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_deployer.email}"
}

resource "google_project_iam_member" "github_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.github_deployer.email}"
}

resource "google_service_account_iam_member" "github_api_user" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.github_deployer.email}"
}

resource "google_service_account_iam_member" "github_ingestion_user" {
  service_account_id = google_service_account.ingestion.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.github_deployer.email}"
}
