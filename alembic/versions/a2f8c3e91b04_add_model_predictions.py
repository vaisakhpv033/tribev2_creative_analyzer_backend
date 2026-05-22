"""Add model prediction columns to neural_scores

Revision ID: a2f8c3e91b04
Revises: 1d30e87810fa
Create Date: 2026-05-21

Adds columns for:
  - XGBoost CTR predictions (predicted_ctr, predicted_class, predicted_proba, prediction_tier)
  - Quantile confidence bounds (ctr_lower_bound, ctr_upper_bound)
  - Raw model feature values for audit trail
    (longest_sustained_above_mean, orbital_mean, visual_std,
     insula_short_mean, attention_onset_second)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a2f8c3e91b04'
down_revision: Union[str, Sequence[str], None] = '1d30e87810fa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add model-based prediction columns to neural_scores."""

    # Model predictions
    op.add_column('neural_scores', sa.Column('predicted_ctr', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('predicted_class', sa.String(), nullable=True))
    op.add_column('neural_scores', sa.Column('predicted_proba', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('prediction_tier', sa.String(), nullable=True))

    # Quantile confidence bounds
    op.add_column('neural_scores', sa.Column('ctr_lower_bound', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('ctr_upper_bound', sa.Float(), nullable=True))

    # Raw model feature values (audit trail)
    op.add_column('neural_scores', sa.Column('longest_sustained_above_mean', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('orbital_mean', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('visual_std', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('insula_short_mean', sa.Float(), nullable=True))
    op.add_column('neural_scores', sa.Column('attention_onset_second', sa.Float(), nullable=True))


def downgrade() -> None:
    """Remove model prediction columns from neural_scores."""

    op.drop_column('neural_scores', 'attention_onset_second')
    op.drop_column('neural_scores', 'insula_short_mean')
    op.drop_column('neural_scores', 'visual_std')
    op.drop_column('neural_scores', 'orbital_mean')
    op.drop_column('neural_scores', 'longest_sustained_above_mean')
    op.drop_column('neural_scores', 'ctr_upper_bound')
    op.drop_column('neural_scores', 'ctr_lower_bound')
    op.drop_column('neural_scores', 'prediction_tier')
    op.drop_column('neural_scores', 'predicted_proba')
    op.drop_column('neural_scores', 'predicted_class')
    op.drop_column('neural_scores', 'predicted_ctr')
