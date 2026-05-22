"""Add creative_insights table

Revision ID: b5d2a4f19c07
Revises: a2f8c3e91b04
Create Date: 2026-05-22

Creates the creative_insights table for storing AI-generated
structured analysis produced by the Gemini LLM.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5d2a4f19c07'
down_revision: Union[str, Sequence[str], None] = 'a2f8c3e91b04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create creative_insights table."""
    op.create_table(
        'creative_insights',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('video_id', sa.UUID(), nullable=True),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('strengths', sa.Text(), nullable=True),
        sa.Column('weaknesses', sa.Text(), nullable=True),
        sa.Column('recommendations', sa.Text(), nullable=True),
        sa.Column('feature_analysis', sa.Text(), nullable=True),
        sa.Column('model_used', sa.String(), nullable=True),
        sa.Column('generated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_creative_insights_video_id'),
        'creative_insights',
        ['video_id'],
        unique=False,
    )


def downgrade() -> None:
    """Drop creative_insights table."""
    op.drop_index(op.f('ix_creative_insights_video_id'), table_name='creative_insights')
    op.drop_table('creative_insights')
