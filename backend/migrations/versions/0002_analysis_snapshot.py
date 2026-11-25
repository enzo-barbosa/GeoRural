"""Guarda o polígono, o nome e a área de cada análise.

Revision ID: 0002_analysis_snapshot
Revises: 0001_initial_schema
Create Date: 2026-11-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_analysis_snapshot"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "analyses", sa.Column("name", sa.String(length=160), nullable=True)
    )
    op.add_column("analyses", sa.Column("geojson", sa.JSON(), nullable=True))
    op.add_column("analyses", sa.Column("area_ha", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("analyses", "area_ha")
    op.drop_column("analyses", "geojson")
    op.drop_column("analyses", "name")
