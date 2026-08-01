import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, LiteralString, Self

import dotenv
import psycopg
from psycopg import sql

SCHEMA_VERSION = "8afea0bd19cc"  # WARN: update this as schema changes


def _get_env_or_user(key: str) -> str:
    if (database_url := os.environ.get(key)) is not None:
        return database_url
    return input(f"{key}? ")


class DBCredentials:
    """Database credentials, fetched from the environment.

    Properties:
        DATABASE_USER
        DATABASE_PASS
        DATABASE_HOST
        DATABASE_PORT
        DATABASE_NAME
    """

    def __init__(self):
        """Create new DB credentials instance."""
        dotenv.load_dotenv()

        self.user: str = _get_env_or_user("DATABASE_USER")
        self.pasw: str = _get_env_or_user("DATABASE_PASS")
        self.host: str = _get_env_or_user("DATABASE_HOST")
        self.port: str = _get_env_or_user("DATABASE_PORT")
        self.name: str = _get_env_or_user("DATABASE_NAME")

    def connect(self, autocommit: bool = True, connect_timeout: int = 10):
        """Open a connection.

        `connect_timeout` matters more than it looks: without it libpq waits on the
        TCP handshake indefinitely, so pointing at a host that is down (or a port
        nothing listens on) hangs the process rather than raising. libpq clamps the
        value to a 2 second minimum.
        """
        return psycopg.connect(
            dbname=self.name,
            user=self.user,
            password=self.pasw,
            host=self.host,
            port=self.port,
            autocommit=autocommit,
            connect_timeout=connect_timeout,
        )


@dataclass
class HologramDetail:
    """Dataclass containing hologram data."""

    recording_session_id: int
    relative_path: Path
    z_depth_um: float
    sha256: bytes

    def as_tuple(self):
        """Return HologramDetail fields as a tuple."""
        # psycopg has no adapter for Path, so the path is rendered here.
        return (
            self.recording_session_id,
            self.relative_path.as_posix(),
            self.z_depth_um,
            self.sha256,
        )


class HolodDatabase:
    """Stores and operates on an Alembic managed database for Holod styled data."""

    def __init__(self):
        """Create a new HolodDatabase instance with a new connection to the database."""
        self.version: str = SCHEMA_VERSION
        self.creds: DBCredentials = DBCredentials()
        # One connection for the life of the instance. autocommit=True makes each
        # statement durable on its own; use `transaction()` to group several.
        self.conn = self.creds.connect()

        # If the guard rejects the database, close before propagating so the failed
        # instance does not leave a socket open.
        try:
            self._check_schema_version()
        except BaseException:
            self.conn.close()
            raise

    def _check_schema_version(self):
        cursor = self.conn.cursor()
        schema_version = cursor.execute("SELECT version_num FROM alembic_version").fetchone()
        if schema_version is None:
            raise Exception("Database does not contain a `version_num` in `alembic_version` table!")
        if schema_version[0] != self.version:
            raise Exception(
                f"database version={schema_version[0]} != python class version={self.version} "
                "run `make db-migrate` or bump the python database version"
            )

    def close(self):
        """Close the connection. Idempotent, so it is safe on an already-closed instance."""
        self.conn.close()

    def __enter__(self) -> Self:
        """Enter a `with` block; the connection is already open."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ):
        """Close the connection on leaving a `with` block, error or not."""
        self.close()

    def transaction(self, force_rollback: bool = False):
        """Group several statements into one atomic unit.

        Under autocommit each statement commits on its own, which is the right default for
        one-off writes but wrong for a multi-table insert. Wrapping those in this context
        manager emits an explicit BEGIN/COMMIT, so a failure part-way rolls the whole
        group back:

            with db.transaction():
                ds = db.register_dataset(...)
                rs = db.insert_recording_session(ds, ...)
                db.insert_hologram([...])
        """
        return self.conn.transaction(force_rollback=force_rollback)

    def _execute(self, stmt: LiteralString, data: tuple[Any, ...]):
        cursor = self.conn.cursor()
        cursor.execute(stmt, data)

    def _execute_returning_id(self, stmt: LiteralString, data: tuple[Any, ...]) -> int:
        """Run an INSERT ... RETURNING id and hand back the generated key."""
        cursor = self.conn.cursor()
        row = cursor.execute(stmt, data).fetchone()
        if row is None:
            raise Exception(f"expected a RETURNING row, statement produced none: {stmt}")
        return row[0]

    def _execute_queries(self, query: sql.SQL | list[sql.SQL]):
        if not isinstance(query, list):
            query = [query]
        queries: list[sql.SQL] = query

        cursor = self.conn.cursor()
        for query in queries:
            cursor.execute(query)

    def register_dataset(
        self, name: str, root_path: Path, created_at: datetime | None = None
    ) -> int:
        """Register a new dataset to the Holod database."""
        # created_at has a NOT NULL DEFAULT now(); COALESCE reproduces it for the None case.
        # `name` is UNIQUE, so re-registering an existing dataset refreshes the path rather
        # than raising.
        stmt = (
            "INSERT INTO dataset (name, root_path, created_at) "
            "VALUES (%s, %s, COALESCE(%s::timestamptz, now())) "
            "ON CONFLICT (name) DO UPDATE SET root_path = EXCLUDED.root_path "
            "RETURNING id;"
        )
        return self._execute_returning_id(stmt, (name, root_path.as_posix(), created_at))

    def insert_recording_session(
        self,
        dataset_id: int,
        wavelength_mm: float,
        l_distance_mm: float,
        pixel_pitch_mm: float,
        recorded_at: datetime | None = None,
    ) -> int:
        """Insert a new recording session row."""
        # recorded_at is nullable with no default, so None can be passed through as NULL.
        stmt = (
            "INSERT INTO recording_session "
            "(dataset_id, wavelength_mm, l_distance_mm, pixel_pitch_mm, recorded_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "RETURNING id;"
        )
        return self._execute_returning_id(
            stmt, (dataset_id, wavelength_mm, l_distance_mm, pixel_pitch_mm, recorded_at)
        )

    def insert_hologram(self, hologram_details: list[HologramDetail]) -> list[int]:
        """Batch-insert holograms, returning the generated ids in input order."""
        if not hologram_details:
            return []

        stmt = (
            "INSERT INTO hologram (recording_session_id, relative_path, z_depth_um, sha256) "
            "VALUES (%s, %s, %s, %s) "
            "RETURNING id;"
        )
        cur = self.conn.cursor()
        holo_tuples = [h.as_tuple() for h in hologram_details]
        # `returning=True` keeps each statement's result set; they are walked with nextset().
        cur.executemany(stmt, holo_tuples, returning=True)

        ids: list[int] = []
        while True:
            row = cur.fetchone()
            if row is not None:
                ids.append(row[0])
            if not cur.nextset():
                break
        return ids

    def insert_run(
        self,
        git_commit: str,
        config_hash: str,
        config: str,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        status: str | None = None,
    ) -> int:
        """Insert a new training run row."""
        # started_at and status have NOT NULL DEFAULTs; COALESCE reproduces them for the
        # None cases. finished_at is nullable with no default, so None passes through.
        stmt = (
            "INSERT INTO run (git_commit, config_hash, config, started_at, finished_at, status) "
            "VALUES (%s, %s, %s, COALESCE(%s::timestamptz, now()), %s, "
            "COALESCE(%s::text, 'running')) "
            "RETURNING id;"
        )
        return self._execute_returning_id(
            stmt, (git_commit, config_hash, config, started_at, finished_at, status)
        )

    def insert_prediction(
        self,
        run_id: int,
        hologram_id: int,
        epoch: int,
        predicted_z_mm: float,
        focus_score: float | None = None,
    ):
        """Insert a new prediction session row."""
        # (run_id, hologram_id, epoch) is the PK, so there is nothing to return.
        # focus_score is nullable with no default, so None can be passed through as NULL.
        stmt = (
            "INSERT INTO prediction (run_id, hologram_id, epoch, predicted_z_mm, focus_score) "
            "VALUES (%s, %s, %s, %s, %s);"
        )
        self._execute(stmt, (run_id, hologram_id, epoch, predicted_z_mm, focus_score))
