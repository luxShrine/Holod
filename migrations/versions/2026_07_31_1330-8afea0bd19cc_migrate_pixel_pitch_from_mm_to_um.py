"""migrate pixel_pitch from mm to um.

Revision ID: 8afea0bd19cc
Revises: 2398653793f2
Create Date: 2026-07-31 13:30:48.979903

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8afea0bd19cc"
down_revision: str | Sequence[str] | None = "2398653793f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE hologram RENAME COLUMN z_depth_mm TO z_depth_um")
    op.execute("UPDATE hologram SET z_depth_um = z_depth_um * 1000")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("UPDATE hologram SET z_depth_um = z_depth_um / 1000")
    op.execute("ALTER TABLE hologram RENAME COLUMN z_depth_um TO z_depth_mm")
