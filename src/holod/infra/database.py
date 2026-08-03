import os
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from types import TracebackType
from typing import Any, LiteralString, NewType, Self

import dotenv
import psycopg
from psycopg import sql
from psycopg.rows import RowFactory, class_row

SCHEMA_VERSION = "8afea0bd19cc"  # WARN: update this as schema changes


def _get_env_or_user(key: str) -> str:
    """Read `key` from the environment, falling back to an interactive prompt."""
    if (database_url := os.environ.get(key)) is not None:
        return database_url
    return input(f"{key}? ")


def _convert_dict_jsonb(config: dict):
    """Wrap a dict so psycopg adapts it to a `jsonb` column instead of a text literal."""
    return psycopg.types.json.Jsonb(config)  # pyright: ignore[reportAttributeAccessIssue]


DSId = NewType("DSId", int)
RSId = NewType("RSId", int)
HoloId = NewType("HoloId", int)
RunId = NewType("RunId", int)


# -- output types ------


@dataclass
class DatasetRow:
    """One row of the `dataset` table: a named collection of holograms on disk."""

    id: DSId
    name: str
    root_path: str
    created_at: datetime

    def __iter__(self):
        """Yield the field values in column order, so the row unpacks like a tuple."""
        yield from (asdict(self).values())


@dataclass
class RecordingRow:
    """One row of the `recording_session` table: the optical setup a hologram was shot with."""

    id: RSId
    dataset_id: DSId
    wavelength_mm: float
    l_distance_mm: float
    pixel_pitch_mm: float
    recorded_at: datetime | None

    def __iter__(self):
        """Yield the field values in column order, so the row unpacks like a tuple."""
        yield from (asdict(self).values())


@dataclass
class HologramRow:
    """One row of the `hologram` table: a single image plus its ground-truth depth."""

    id: HoloId
    recording_session_id: RSId
    relative_path: str
    z_depth_um: float
    sha256: bytes

    def __iter__(self):
        """Yield the field values in column order, so the row unpacks like a tuple."""
        yield from (asdict(self).values())


@dataclass
class RunRow:
    """One row of the `run` table: a single training run and the config it ran with."""

    id: RunId
    git_commit: str
    config_hash: str
    config: str
    started_at: datetime
    finished_at: datetime
    status: str

    def __iter__(self):
        """Yield the field values in column order, so the row unpacks like a tuple."""
        yield from (asdict(self).values())


@dataclass
class PredictionRow:
    """One row of the `prediction` table: what a run predicted for a hologram at an epoch."""

    run_id: RunId
    hologram_id: HoloId
    epoch: int
    predicted_z_mm: float
    focus_score: float

    def __iter__(self):
        """Yield the field values in column order, so the row unpacks like a tuple."""
        yield from (asdict(self).values())

    @property
    def id(self):
        """Return the PK of prediction row."""
        return (self.run_id, self.hologram_id, self.epoch)


TableRow = DatasetRow | RecordingRow | HologramRow | RunRow | PredictionRow
# results of selecting rows can be tuples of values if columns are selected, or TableRows
SQLTypes = str | bytes | dict | datetime | float | None | int


class Tables(StrEnum):
    """Tables of the Holod schema; each member's value is the table name in Postgres."""

    Dataset = auto()
    Recording_Session = auto()
    Hologram = auto()
    Run = auto()
    Prediction = auto()

    def get_row_factory(self) -> RowFactory[TableRow]:
        """Return the psycopg row factory that maps this table's rows to its dataclass."""
        match self:
            case Tables.Dataset:
                return class_row(DatasetRow)
            case Tables.Recording_Session:
                return class_row(RecordingRow)
            case Tables.Hologram:
                return class_row(HologramRow)
            case Tables.Run:
                return class_row(RunRow)
            case Tables.Prediction:
                return class_row(PredictionRow)


# -- ------


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

    recording_session_id: RSId
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
        """Raise unless the database's Alembic revision matches `SCHEMA_VERSION`."""
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

    def _execute(
        self,
        stmt: LiteralString | sql.SQL | sql.Composed,
        data: tuple[Any, ...] | None = None,
        row_factory: RowFactory[TableRow] | None = None,
    ):
        """Execute one statement and return its cursor, optionally mapping rows to a dataclass."""
        cursor = (
            self.conn.cursor() if row_factory is None else self.conn.cursor(row_factory=row_factory)
        )
        return cursor.execute(stmt, data)

    def _execute_returning_id(self, stmt: LiteralString, data: tuple[Any, ...]) -> int:
        """Run an INSERT ... RETURNING id and hand back the generated key."""
        cursor = self.conn.cursor()
        row = cursor.execute(stmt, data).fetchone()
        if row is None:
            raise Exception(f"expected a RETURNING row, statement produced none: {stmt}")
        return row[0]

    def _execute_queries(self, query: sql.SQL | list[sql.SQL]):
        """Run one or more parameterless queries in order on a single cursor."""
        if not isinstance(query, list):
            query = [query]
        queries: list[sql.SQL] = query

        cursor = self.conn.cursor()
        for query in queries:
            cursor.execute(query)

    def register_dataset(
        self, name: str, root_path: Path, created_at: datetime | None = None
    ) -> DSId:
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
        return DSId(self._execute_returning_id(stmt, (name, root_path.as_posix(), created_at)))

    def insert_recording_session(
        self,
        dataset_id: DSId,
        wavelength_mm: float,
        l_distance_mm: float,
        pixel_pitch_mm: float,
        recorded_at: datetime | None = None,
    ) -> RSId:
        """Insert a new recording session row."""
        # recorded_at is nullable with no default, so None can be passed through as NULL.
        stmt = (
            "INSERT INTO recording_session "
            "(dataset_id, wavelength_mm, l_distance_mm, pixel_pitch_mm, recorded_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "RETURNING id;"
        )
        return RSId(
            self._execute_returning_id(
                stmt, (dataset_id, wavelength_mm, l_distance_mm, pixel_pitch_mm, recorded_at)
            )
        )

    def insert_hologram(self, hologram_details: list[HologramDetail]) -> list[HoloId]:
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
        # ensure that regardless of executemany internals that this is done as one batch
        with self.transaction():
            # `returning=True` keeps each statement's result set; they are walked with nextset().
            cur.executemany(stmt, holo_tuples, returning=True)

        ids: list[HoloId] = []
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
        config: dict,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        status: str | None = None,
    ) -> RunId:
        """Insert a new training run row."""
        json_config = _convert_dict_jsonb(config)
        # started_at and status have NOT NULL DEFAULTs; COALESCE reproduces them for the
        # None cases. finished_at is nullable with no default, so None passes through.
        stmt = (
            "INSERT INTO run (git_commit, config_hash, config, started_at, finished_at, status) "
            "VALUES (%s, %s, %s, COALESCE(%s::timestamptz, now()), %s, "
            "COALESCE(%s::text, 'running')) "
            "RETURNING id;"
        )
        return RunId(
            self._execute_returning_id(
                stmt, (git_commit, config_hash, json_config, started_at, finished_at, status)
            )
        )

    def insert_prediction(
        self,
        run_id: RunId,
        hologram_id: HoloId,
        epoch: int,
        predicted_z_mm: float,
        focus_score: float | None = None,
    ):
        """Insert a new prediction session row.

        (run_id, hologram_id, epoch) is the PK, so there is nothing to return.
        """
        # focus_score is nullable with no default, so None can be passed through as NULL.
        stmt = (
            "INSERT INTO prediction (run_id, hologram_id, epoch, predicted_z_mm, focus_score) "
            "VALUES (%s, %s, %s, %s, %s);"
        )
        self._execute(stmt, (run_id, hologram_id, epoch, predicted_z_mm, focus_score))

    def _select(
        self,
        table: Tables,
        column: str | None,
        condition: tuple[str, Any] | None,
        amount: int,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Run `SELECT <column> FROM <table> [WHERE <field> = <value>] LIMIT <amount>`."""
        if amount < 0:
            raise ValueError(f"amount must not be negative, got {amount}")

        # TODO: if no column, then we should use the row factory from each dataclass per database
        if column is None:
            row_factory = table.get_row_factory()
            column_sql = sql.SQL("*")
        else:
            row_factory = None
            column_sql = sql.Identifier(column)

        tbl_sql = sql.Identifier(table.lower())

        if condition is None:
            stmt = sql.SQL("SELECT {column} FROM {tbl} LIMIT %s;").format(
                column=column_sql, tbl=tbl_sql
            )
            return self._execute(stmt, (amount,), row_factory).fetchall()

        (field, value) = condition
        # `= NULL` is never true, not even for a NULL column, so a None here has to
        # become IS NULL or the query silently returns nothing.
        comparison = sql.SQL("IS NULL") if value is None else sql.SQL("= %s")
        stmt = sql.SQL("SELECT {column} FROM {tbl} WHERE {field} {comparison} LIMIT %s;").format(
            column=column_sql, tbl=tbl_sql, field=sql.Identifier(field), comparison=comparison
        )
        data = (amount,) if value is None else (value, amount)
        return self._execute(stmt, data, row_factory).fetchall()

    def select_dataset(
        self,
        column: str | None = None,
        condition: tuple[str, Any] | None = None,
        amount: int = 50,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Select datasets from the Holod database.

        `column` defaults to every column; `condition` is a (column, value) pair
        filtering the rows, and `amount` caps how many come back.
        """
        return self._select(Tables("dataset"), column, condition, amount)

    def select_recording_session(
        self,
        column: str | None = None,
        condition: tuple[str, Any] | None = None,
        amount: int = 50,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Select recording sessions from the Holod database."""
        return self._select(Tables("recording_session"), column, condition, amount)

    def select_holograms(
        self,
        column: str | None = None,
        condition: tuple[str, Any] | None = None,
        amount: int = 50,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Select holograms from the Holod database."""
        return self._select(Tables("hologram"), column, condition, amount)

    def select_run(
        self,
        column: str | None = None,
        condition: tuple[str, Any] | None = None,
        amount: int = 50,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Select runs from the Holod database."""
        return self._select(Tables("run"), column, condition, amount)

    def select_prediction(
        self,
        column: str | None = None,
        condition: tuple[str, Any] | None = None,
        amount: int = 50,
    ) -> list[tuple[SQLTypes, ...]] | list[TableRow]:
        """Select predictions from the Holod database."""
        return self._select(Tables("prediction"), column, condition, amount)
