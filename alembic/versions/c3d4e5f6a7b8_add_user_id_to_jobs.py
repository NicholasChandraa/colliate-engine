"""add user_id to jobs

Revision ID: c3d4e5f6a7b8
Revises: 9e22c96d63e2
Create Date: 2026-03-07

"""
from alembic import op
import sqlalchemy as sa

revision = 'c3d4e5f6a7b8'
down_revision = '9e22c96d63e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('jobs', sa.Column('user_id', sa.String(36), nullable=True))
    op.create_index('ix_jobs_user_id', 'jobs', ['user_id'])


def downgrade() -> None:
    op.drop_index('ix_jobs_user_id', table_name='jobs')
    op.drop_column('jobs', 'user_id')
