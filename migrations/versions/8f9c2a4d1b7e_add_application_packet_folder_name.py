"""Add application packet folder name

Revision ID: 8f9c2a4d1b7e
Revises: 055aec76ff62
Create Date: 2026-07-22 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "8f9c2a4d1b7e"
down_revision = "055aec76ff62"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "job_scores",
        sa.Column(
            "application_packet_folder_name",
            sa.String(length=20),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("job_scores", "application_packet_folder_name")
