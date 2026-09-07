# ElderHelp Google Cloud infrastructure

This Terraform module creates the staging API, ingestion job, PostgreSQL database, private report
bucket, service identities, Artifact Registry repository, required APIs, and GitHub workload
identity provider.

Provide an immutable container image digest and pass a database password outside version control:

```sh
export TF_VAR_database_password='use-a-secret-generated-value'
tofu init
tofu plan -var='project_id=YOUR_PROJECT' \
  -var='container_image=REGION-docker.pkg.dev/PROJECT/elderhelp/api@sha256:DIGEST'
tofu apply -var='project_id=YOUR_PROJECT' \
  -var='container_image=REGION-docker.pkg.dev/PROJECT/elderhelp/api@sha256:DIGEST'
```

The module creates the application database user and a Secret Manager version containing a Cloud
SQL Unix-socket URL with this shape:

```text
postgresql+asyncpg://USER:PASSWORD@/elderhelp?host=/cloudsql/PROJECT:REGION:INSTANCE
```

Apply migrations before the first ingestion job. Terraform creates no public PDF permissions. The
API is not granted storage read access because it returns publisher links and approved excerpts
from PostgreSQL.

The password is sensitive but is necessarily present in Terraform state. Use encrypted remote
state with restricted access for shared environments. To bootstrap a new project, first target the
required APIs and Artifact Registry repository, push the initial image, and then apply the complete
module with the immutable digest.

After apply, copy the `github_workload_identity_provider` and
`github_deployer_service_account` outputs into GitHub environment variables named
`GCP_WORKLOAD_IDENTITY_PROVIDER` and `GCP_DEPLOYER_SERVICE_ACCOUNT`. Add `GCP_PROJECT_ID` as a
third staging environment variable. The manual `Deploy staging` workflow then builds an immutable
commit-tagged image, updates the API and both jobs, and runs the migration job. Run ingestion only
after the curated manifest and report rights have been reviewed.
