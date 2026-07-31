"""initial schema.

Revision ID: 2398653793f2
Revises:
Create Date: 2026-07-31 12:48:12.267979

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2398653793f2"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("""
        CREATE TABLE dataset (
            id          INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
            name        TEXT NOT NULL UNIQUE,
            root_path   TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)

    op.execute("""
        CREATE TABLE recording_session (
            id              INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
            dataset_id      INTEGER NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
            wavelength_mm   DOUBLE PRECISION NOT NULL,
            l_distance_mm   DOUBLE PRECISION NOT NULL,
            pixel_pitch_mm  DOUBLE PRECISION NOT NULL,
            recorded_at     TIMESTAMPTZ
        );
        CREATE INDEX ON recording_session (dataset_id);
    """)

    op.execute("""
        CREATE TABLE hologram (
            id                    BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
            recording_session_id  INTEGER NOT NULL REFERENCES recording_session(id) ON DELETE CASCADE,
            relative_path         TEXT NOT NULL,
            z_depth_mm            DOUBLE PRECISION NOT NULL,
            sha256                BYTEA NOT NULL CHECK (octet_length(sha256) = 32),
            UNIQUE (recording_session_id, relative_path)
        );
        CREATE INDEX ON hologram (sha256);
    """)

    op.execute("""
        CREATE TABLE run (
            id           INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
            git_commit   CHAR(40) NOT NULL,
            config_hash  CHAR(64) NOT NULL,
            config       JSONB NOT NULL,
            started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at  TIMESTAMPTZ,
            status       TEXT NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'completed', 'failed'))
        );
    """)

    op.execute("""
        CREATE TABLE prediction (
            run_id          INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
            hologram_id     BIGINT  NOT NULL REFERENCES hologram(id) ON DELETE CASCADE,
            epoch           INTEGER NOT NULL,
            predicted_z_mm  DOUBLE PRECISION NOT NULL,
            focus_score     DOUBLE PRECISION,
            PRIMARY KEY (run_id, hologram_id, epoch)
        );
        CREATE INDEX ON prediction (hologram_id);
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TABLE IF EXISTS prediction")
    op.execute("DROP TABLE IF EXISTS run")
    op.execute("DROP TABLE IF EXISTS hologram")
    op.execute("DROP TABLE IF EXISTS recording_session")
    op.execute("DROP TABLE IF EXISTS dataset")
