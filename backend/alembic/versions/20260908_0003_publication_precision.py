"""Record publication precision instead of presenting unknown month/day as a fact."""

from alembic import op

revision = "20260908_0003"
down_revision = "20260908_0002"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE reports ADD COLUMN publication_precision "
        "VARCHAR(12) NOT NULL DEFAULT 'unknown'"
    )


def downgrade():
    op.drop_column("reports", "publication_precision")
