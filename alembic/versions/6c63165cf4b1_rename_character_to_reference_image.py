"""rename_character_to_reference_image

Revision ID: 6c63165cf4b1
Revises: c1017d84183b
Create Date: 2026-03-07 16:11:18.963779

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c63165cf4b1'
down_revision: Union[str, Sequence[str], None] = 'c1017d84183b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("jobs", "character_image_path", new_column_name="reference_image_path")
    op.add_column("jobs", sa.Column("reference_image_type", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "reference_image_type")
    op.alter_column("jobs", "reference_image_path", new_column_name="character_image_path")
