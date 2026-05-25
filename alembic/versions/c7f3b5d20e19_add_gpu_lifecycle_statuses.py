"""Add GPU lifecycle status values to JobStatus enum

Revision ID: c7f3b5d20e19
Revises: b5d2a4f19c07
Create Date: 2026-05-25

Adds PROVISIONING_GPU and BOOTING_GPU values to the jobstatus
PostgreSQL enum type for RunPod pod lifecycle tracking.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c7f3b5d20e19'
down_revision: Union[str, Sequence[str], None] = 'b5d2a4f19c07'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new GPU lifecycle status values to the jobstatus enum."""
    # PostgreSQL requires ALTER TYPE to add new enum values.
    # These cannot be done inside a transaction, so we commit first.
    op.execute("COMMIT")
    op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'PROVISIONING_GPU'")
    op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'BOOTING_GPU'")


def downgrade() -> None:
    """Remove GPU lifecycle status values from the jobstatus enum.

    NOTE: PostgreSQL does not support removing enum values directly.
    This downgrade recreates the enum type without the new values.
    Any rows with PROVISIONING_GPU or BOOTING_GPU will be set to PENDING.
    """
    # Update any rows using the old statuses
    op.execute("""
        UPDATE videos SET status = 'PENDING'
        WHERE status IN ('PROVISIONING_GPU', 'BOOTING_GPU')
    """)

    # Recreate the enum without the new values
    op.execute("ALTER TYPE jobstatus RENAME TO jobstatus_old")
    op.execute("""
        CREATE TYPE jobstatus AS ENUM (
            'PENDING', 'UPLOADING', 'INFERENCE', 'ANALYZING', 'COMPLETED', 'FAILED'
        )
    """)
    op.execute("""
        ALTER TABLE videos
        ALTER COLUMN status TYPE jobstatus USING status::text::jobstatus
    """)
    op.execute("DROP TYPE jobstatus_old")
