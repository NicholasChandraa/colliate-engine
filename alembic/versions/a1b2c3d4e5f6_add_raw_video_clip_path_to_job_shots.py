"""add_raw_video_clip_path_to_job_shots

Revision ID: a1b2c3d4e5f6
Revises: 3bcf0cfdde97
Create Date: 2026-03-08 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '3bcf0cfdde97'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('job_shots', sa.Column('raw_video_clip_path', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('job_shots', 'raw_video_clip_path')
