"""Tests for the Postgres layer in ``holod.infra.database``.

Covers four failure classes that only appear once a statement actually reaches
Postgres, and so are invisible to ruff/mypy:

* **schema mismatch** -- a column the SQL names no longer exists, or
  ``SCHEMA_VERSION`` has drifted from the alembic head;
* **constraint behaviour** -- which duplicates are refreshed (``dataset.name``)
  and which raise (``hologram`` paths, ``prediction`` keys), plus the CHECK and
  FK guards;
* **path round-tripping** -- a ``Path`` must reach the database as a POSIX
  string and come back reconstructible, so a dataset ingested on Windows is
  readable on Linux;
* **COALESCE drift** -- the fallbacks written into the INSERTs must keep
  matching the column ``DEFAULT``s they duplicate.

These need a live Postgres. Bring one up with ``make db-up && make db-migrate``,
then ``make test-db``. Without a reachable server every test here skips, unless
``HOLOD_REQUIRE_DB=1`` is set (CI does) -- there a skip is a failure.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString

import psycopg
import pytest
from psycopg import sql

from holod.infra import database
from holod.infra.database import (
    SCHEMA_VERSION,
    DatasetRow,
    DBCredentials,
    DSId,
    HolodDatabase,
    HologramDetail,
    HologramRow,
    PredictionRow,
    RecordingRow,
    RSId,
    RunRow,
    SQLTypes,
    TableRow,
    Tables,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

GIT_COMMIT = "a" * 40  # run.git_commit is CHAR(40); short values would be blank-padded
CONFIG = {"lr": 0.0001}  # run.config is JSONB, so this has to parse as JSON
CONFIG_HASH = database._hash_config(CONFIG)  # run.config_hash is CHAR(64)
DIGEST = bytes(32)  # hologram.sha256 is CHECK (octet_length(sha256) = 32)


def _db_reachable() -> bool:
    """Report whether a Postgres server answers, without running the version guard."""
    # Deliberately not HolodDatabase(): that would raise on a schema mismatch, which
    # would then be misreported as "no database" and silently skip the whole suite.
    # The short timeout matters because this runs at collection time, the default
    # would stall every `make test` on a machine whose configured host is unreachable.
    try:
        DBCredentials().connect(connect_timeout=2).close()
    except Exception:
        return False
    return True


# Skipped locally when no server is running, but an error in CI: there the service
# container means an unreachable database is a broken workflow, not a missing optional
# dependency, and a silent skip would hide that the suite stopped running entirely.
NO_DB = not _db_reachable() and not os.environ.get("HOLOD_REQUIRE_DB")
SKIP_REASON = (
    "no Postgres reachable; run `make db-up && make db-migrate` (or set HOLOD_REQUIRE_DB=1)"
)

requires_db = pytest.mark.skipif(NO_DB, reason=SKIP_REASON)

pytestmark = pytest.mark.db


@pytest.fixture(scope="session")
def db() -> Iterator[HolodDatabase]:
    """Open one connection for the whole module and close it at the end.

    Skipping here rather than through a module-level mark keeps the two tests that
    need no database (the alembic-head check and the Path rendering check) running
    on a machine with no Postgres.
    """
    if NO_DB:
        pytest.skip(SKIP_REASON)
    with HolodDatabase() as database_conn:
        yield database_conn


@pytest.fixture
def clean_db(db: HolodDatabase) -> Iterator[HolodDatabase]:
    """Run one test inside a transaction that is always rolled back.

    Identity sequences do not roll back, so tests must never assert on a specific
    id value, only on relationships between ids.
    """
    with db.transaction():
        yield db
        # Swallowed by transaction(): unwinds the block without escaping to pytest.
        raise psycopg.Rollback


def make_session(db: HolodDatabase, name: str = "test-ds") -> tuple[DSId, RSId]:
    """Create a dataset and a recording session; return both ids."""
    dataset_id = db.register_dataset(name, Path("test_root/"))
    session_id = db.insert_recording_session(dataset_id, 0.000405, 12.0, 0.0038)
    return dataset_id, session_id


def column_rows(db: HolodDatabase, table: str) -> set[tuple[str, str, str]]:
    """Return {(column, type, nullability)} as the live database reports it."""
    rows = db.conn.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s;",
        (table,),
    ).fetchall()
    return {(str(c), str(t), str(n)) for c, t, n in rows}


def one_row(db: HolodDatabase, stmt: LiteralString, data: tuple[Any, ...] = ()) -> Any:
    """Run a query expected to produce exactly one row and return that row."""
    row = db.conn.execute(stmt, data).fetchone()
    assert row is not None, f"expected one row from: {stmt}"
    return row


# -- schema mismatch ------------------------------------------------------------


def test_schema_version_matches_alembic_head() -> None:
    """SCHEMA_VERSION must equal the newest revision on disk."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    repo_root = Path(__file__).resolve().parents[2].absolute()

    head = ScriptDirectory.from_config(Config(str(repo_root / "alembic.ini"))).get_current_head()
    assert head == SCHEMA_VERSION, (
        f"SCHEMA_VERSION={SCHEMA_VERSION} but the alembic head is {head}; "
        "bump SCHEMA_VERSION in database.py when adding a migration"
    )


def test_alembic_version_matches_schema_version(db: HolodDatabase) -> None:
    """The migrated database reports the revision the Python class expects."""
    (version,) = one_row(db, "SELECT version_num FROM alembic_version;")
    assert version == SCHEMA_VERSION


@requires_db
def test_version_guard_rejects_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wrong schema version raises and does not leak the connection it opened."""
    opened: list[psycopg.Connection[Any]] = []
    real_connect = DBCredentials.connect

    def spy(self: DBCredentials, autocommit: bool = True) -> psycopg.Connection[Any]:
        conn = real_connect(self, autocommit)
        opened.append(conn)
        return conn

    monkeypatch.setattr(DBCredentials, "connect", spy)
    monkeypatch.setattr(database, "SCHEMA_VERSION", "deadbeefcafe")

    with pytest.raises(Exception, match="deadbeefcafe"):
        HolodDatabase()

    assert opened, "the guard should have opened a connection before rejecting"
    assert opened[-1].closed, "a rejected HolodDatabase must not leave a socket open"


def test_table_columns_match_expected(db: HolodDatabase) -> None:
    """Every column the INSERTs name exists, with the expected type and nullability."""
    expected = {
        "dataset": {
            ("id", "integer", "NO"),
            ("name", "text", "NO"),
            ("root_path", "text", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
        },
        "recording_session": {
            ("id", "integer", "NO"),
            ("dataset_id", "integer", "NO"),
            ("wavelength_mm", "double precision", "NO"),
            ("l_distance_mm", "double precision", "NO"),
            ("pixel_pitch_mm", "double precision", "NO"),
            ("recorded_at", "timestamp with time zone", "YES"),
        },
        "hologram": {
            ("id", "bigint", "NO"),
            ("recording_session_id", "integer", "NO"),
            ("relative_path", "text", "NO"),
            ("z_depth_um", "double precision", "NO"),
            ("sha256", "bytea", "NO"),
        },
        "run": {
            ("id", "integer", "NO"),
            ("git_commit", "character", "NO"),
            ("config_hash", "character", "NO"),
            ("config", "jsonb", "NO"),
            ("started_at", "timestamp with time zone", "NO"),
            ("finished_at", "timestamp with time zone", "YES"),
            ("status", "text", "NO"),
        },
        "prediction": {
            ("run_id", "integer", "NO"),
            ("hologram_id", "bigint", "NO"),
            ("epoch", "integer", "NO"),
            ("predicted_z_mm", "double precision", "NO"),
            ("focus_score", "double precision", "YES"),
        },
    }
    for table, columns in expected.items():
        assert column_rows(db, table) == columns, f"{table} has drifted from the expected schema"


def test_every_insert_reaches_the_database(clean_db: HolodDatabase) -> None:
    """Each insert method executes against the live schema.

    Complements the information_schema comparison: that inspects the database,
    this exercises the SQL text, so a typo in a column name fails here.
    """
    dataset_id, session_id = make_session(clean_db)
    holo_ids = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("img/0001.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(run_id, holo_ids[0], epoch=0, predicted_z_mm=1.5)

    assert dataset_id > 0
    assert session_id > 0
    assert len(holo_ids) == 1
    assert run_id > 0


# -- unique / check / foreign-key behaviour -------------------------------------


def test_register_dataset_is_idempotent(clean_db: HolodDatabase) -> None:
    """Registering the same dataset name twice returns the same id."""
    first = clean_db.register_dataset("dupe-ds", Path("a/"))
    second = clean_db.register_dataset("dupe-ds", Path("a/"))
    assert first == second


def test_register_dataset_refreshes_root_path(clean_db: HolodDatabase) -> None:
    """Re-registering an existing name overwrites root_path rather than raising."""
    dataset_id = clean_db.register_dataset("moving-ds", Path("before/"))
    clean_db.register_dataset("moving-ds", Path("after/"))

    (root_path,) = one_row(clean_db, "SELECT root_path FROM dataset WHERE id = %s;", (dataset_id,))
    assert root_path == "after"


def test_duplicate_hologram_path_raises(clean_db: HolodDatabase) -> None:
    """Two holograms with the same path in one session violate the UNIQUE constraint."""
    _, session_id = make_session(clean_db, "holo-dupe-ds")
    detail = HologramDetail(session_id, Path("img/same.png"), 1500.0, DIGEST)
    clean_db.insert_hologram([detail])

    # Nested transaction => SAVEPOINT, so the failure does not poison the outer rollback.
    with pytest.raises(psycopg.errors.UniqueViolation), clean_db.transaction():
        clean_db.insert_hologram([detail])


def test_duplicate_prediction_key_raises(clean_db: HolodDatabase) -> None:
    """(run_id, hologram_id, epoch) is the primary key, so it cannot repeat."""
    _, session_id = make_session(clean_db, "pred-dupe-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("img/p.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(run_id, holo_id, epoch=3, predicted_z_mm=1.5)

    with pytest.raises(psycopg.errors.UniqueViolation), clean_db.transaction():
        clean_db.insert_prediction(run_id, holo_id, epoch=3, predicted_z_mm=9.9)


def test_sha256_must_be_32_bytes(clean_db: HolodDatabase) -> None:
    """A digest of the wrong length is rejected.

    Guards the hexdigest()-instead-of-digest() mistake: a hex string reaches
    bytea as 64 bytes, not 32.
    """
    _, session_id = make_session(clean_db, "sha-ds")
    short = HologramDetail(session_id, Path("img/short.png"), 1500.0, bytes(31))

    with pytest.raises(psycopg.errors.CheckViolation), clean_db.transaction():
        clean_db.insert_hologram([short])


def test_invalid_run_status_rejected(clean_db: HolodDatabase) -> None:
    """run.status only accepts running/completed/failed."""
    with pytest.raises(psycopg.errors.CheckViolation), clean_db.transaction():
        clean_db.insert_run(GIT_COMMIT, CONFIG, status="bogus")  # pyright: ignore[reportArgumentType]


def test_orphan_recording_session_rejected(clean_db: HolodDatabase) -> None:
    """A recording session cannot reference a dataset that does not exist."""
    with pytest.raises(psycopg.errors.ForeignKeyViolation), clean_db.transaction():
        clean_db.insert_recording_session(DSId(-1), 0.000405, 12.0, 0.0038)


def test_deleting_dataset_cascades(clean_db: HolodDatabase) -> None:
    """Deleting a dataset removes its sessions, holograms and predictions."""
    dataset_id, session_id = make_session(clean_db, "cascade-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("img/c.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(run_id, holo_id, epoch=0, predicted_z_mm=1.5)

    clean_db.conn.execute("DELETE FROM dataset WHERE id = %s;", (dataset_id,))

    (sessions,) = one_row(
        clean_db, "SELECT count(*) FROM recording_session WHERE id = %s;", (session_id,)
    )
    (holograms,) = one_row(clean_db, "SELECT count(*) FROM hologram WHERE id = %s;", (holo_id,))
    (predictions,) = one_row(
        clean_db, "SELECT count(*) FROM prediction WHERE hologram_id = %s;", (holo_id,)
    )
    assert (sessions, holograms, predictions) == (0, 0, 0)


# -- path round-tripping --------------------------------------------------------


def test_hologram_detail_as_tuple_renders_path() -> None:
    """as_tuple() yields a string, not a Path.

    psycopg has no adapter for pathlib.Path, so passing one through raises
    "cannot adapt type 'WindowsPath'". No database needed.
    """
    detail = HologramDetail(RSId(1), Path("img/0001.png"), 1500.0, DIGEST)
    rendered = detail.as_tuple()[1]
    assert isinstance(rendered, str)
    assert rendered == "img/0001.png"


def test_hologram_path_round_trips(clean_db: HolodDatabase) -> None:
    """A hologram path stores as POSIX text and reconstructs to the same Path."""
    _, session_id = make_session(clean_db, "path-ds")
    original = Path("sub/dir/img.png")
    (holo_id,) = clean_db.insert_hologram([HologramDetail(session_id, original, 1500.0, DIGEST)])

    (stored,) = one_row(clean_db, "SELECT relative_path FROM hologram WHERE id = %s;", (holo_id,))
    assert stored == original.as_posix()
    assert "\\" not in stored, "a backslash here would not resolve on Linux"
    assert Path(stored) == original


def test_dataset_root_path_round_trips(clean_db: HolodDatabase) -> None:
    """A dataset root path stores as POSIX text and reconstructs to the same Path."""
    original = Path("datasets/MW_Dataset")
    dataset_id = clean_db.register_dataset("roundtrip-ds", original)

    (stored,) = one_row(clean_db, "SELECT root_path FROM dataset WHERE id = %s;", (dataset_id,))
    assert stored == original.as_posix()
    assert "\\" not in stored
    assert Path(stored) == original


@pytest.mark.skipif(sys.platform != "win32", reason="backslash paths only parse on Windows")
def test_windows_path_stored_as_posix(clean_db: HolodDatabase) -> None:
    """A Windows-authored path is normalised before it is stored.

    This is the portability guarantee: a dataset ingested on Windows has to be
    readable by a training run on Linux.
    """
    _, session_id = make_session(clean_db, "winpath-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("sub\\dir\\img.png"), 1500.0, DIGEST)]
    )

    (stored,) = one_row(clean_db, "SELECT relative_path FROM hologram WHERE id = %s;", (holo_id,))
    assert stored == "sub/dir/img.png"


# -- COALESCE defaults ----------------------------------------------------------
#
# These never name the default value itself, that would just duplicate the
# duplication. They assert the property instead: passing None produces the same
# row as omitting the column. now() is transaction_timestamp(), constant for a
# whole transaction, so the two timestamps compare exactly equal.


def test_register_dataset_default_created_at(clean_db: HolodDatabase) -> None:
    """created_at=None yields whatever the column DEFAULT would have."""
    expected = one_row(
        clean_db,
        "INSERT INTO dataset (name, root_path) VALUES (%s, %s) RETURNING created_at;",
        ("default-baseline", "a/"),
    )
    dataset_id = clean_db.register_dataset("default-coalesce", Path("a/"))
    actual = one_row(clean_db, "SELECT created_at FROM dataset WHERE id = %s;", (dataset_id,))

    assert actual == expected


def test_insert_run_defaults(clean_db: HolodDatabase) -> None:
    """started_at=None and status=None yield whatever the column DEFAULTs would."""
    jsonb_config = database._convert_dict_jsonb(CONFIG)
    expected = one_row(
        clean_db,
        "INSERT INTO run (git_commit, config_hash, config) VALUES (%s, %s, %s) "
        "RETURNING started_at, status;",
        (GIT_COMMIT, CONFIG_HASH, jsonb_config),
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    actual = one_row(clean_db, "SELECT started_at, status FROM run WHERE id = %s;", (run_id,))

    assert actual == expected


def test_explicit_values_override_defaults(clean_db: HolodDatabase) -> None:
    """The other half of COALESCE: a supplied value is the one stored."""
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG, started_at=moment, status="completed")
    dataset_id = clean_db.register_dataset("explicit-ds", Path("a/"), created_at=moment)

    assert one_row(clean_db, "SELECT started_at, status FROM run WHERE id = %s;", (run_id,)) == (
        moment,
        "completed",
    )
    assert one_row(clean_db, "SELECT created_at FROM dataset WHERE id = %s;", (dataset_id,)) == (
        moment,
    )


def test_nullable_columns_stay_null(clean_db: HolodDatabase) -> None:
    """Columns with no DEFAULT stay NULL when omitted, and must not grow a COALESCE."""
    _, session_id = make_session(clean_db, "nullable-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("img/n.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(run_id, holo_id, epoch=0, predicted_z_mm=1.5)

    assert one_row(
        clean_db, "SELECT recorded_at FROM recording_session WHERE id = %s;", (session_id,)
    ) == (None,)
    assert one_row(clean_db, "SELECT finished_at FROM run WHERE id = %s;", (run_id,)) == (None,)
    assert one_row(
        clean_db, "SELECT focus_score FROM prediction WHERE run_id = %s;", (run_id,)
    ) == (None,)


def test_defaulted_columns_snapshot(db: HolodDatabase) -> None:
    """Exactly these columns carry a DEFAULT.

    The equivalence tests above cannot see a *newly added* default on a column the
    code inserts explicitly; this catches that case.
    """
    # GENERATED ALWAYS AS IDENTITY columns report column_default IS NULL, so the
    # surrogate keys drop out of this query without needing to be filtered by name.
    rows = db.conn.execute(
        "SELECT table_name, column_name, column_default FROM information_schema.columns "
        "WHERE table_schema = 'public' AND column_default IS NOT NULL "
        "AND table_name IN ('dataset', 'recording_session', 'hologram', 'run', 'prediction');"
    ).fetchall()
    defaults = {(str(table), str(column), str(default)) for table, column, default in rows}
    assert defaults == {
        ("dataset", "created_at", "now()"),
        ("run", "started_at", "now()"),
        ("run", "status", "'running'::text"),
    }


# -- foreign-key chain and connection semantics ---------------------------------


def test_full_chain_round_trip(clean_db: HolodDatabase) -> None:
    """Every generated id is usable as the foreign key of the next insert."""
    dataset_id, session_id = make_session(clean_db, "chain-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("img/chain.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(run_id, holo_id, epoch=0, predicted_z_mm=1.5)

    row = one_row(
        clean_db,
        "SELECT d.name, h.relative_path, p.predicted_z_mm, r.status "
        "FROM prediction p "
        "JOIN run r ON r.id = p.run_id "
        "JOIN hologram h ON h.id = p.hologram_id "
        "JOIN recording_session rs ON rs.id = h.recording_session_id "
        "JOIN dataset d ON d.id = rs.dataset_id "
        "WHERE d.id = %s;",
        (dataset_id,),
    )
    assert row == ("chain-ds", "img/chain.png", 1.5, "running")


def test_insert_hologram_returns_ids_in_order(clean_db: HolodDatabase) -> None:
    """Batch insert returns generated ids matching the input order."""
    _, session_id = make_session(clean_db, "order-ds")
    paths = [Path(f"img/{i:04d}.png") for i in range(5)]
    holo_ids = clean_db.insert_hologram(
        [HologramDetail(session_id, p, 1500.0 + i, DIGEST) for i, p in enumerate(paths)]
    )

    assert len(holo_ids) == len(paths)
    assert len(set(holo_ids)) == len(paths), "ids must be distinct"
    stored = [
        one_row(clean_db, "SELECT relative_path FROM hologram WHERE id = %s;", (i,))[0]
        for i in holo_ids
    ]
    assert stored == [p.as_posix() for p in paths]


def test_insert_hologram_empty_list(clean_db: HolodDatabase) -> None:
    """An empty batch is a no-op returning no ids."""
    assert clean_db.insert_hologram([]) == []


def test_transaction_rolls_back_on_error(db: HolodDatabase) -> None:
    """A failure part-way through a transaction discards the whole group."""
    marker = "rollback-marker-ds"
    with pytest.raises(RuntimeError, match="boom"), db.transaction():
        db.register_dataset(marker, Path("a/"))
        raise RuntimeError("boom")

    (count,) = one_row(db, "SELECT count(*) FROM dataset WHERE name = %s;", (marker,))
    assert count == 0, "the dataset written before the error should not have survived"


def test_autocommit_persists_without_transaction(db: HolodDatabase) -> None:
    """A write outside an explicit transaction is visible to other connections.

    Regression test for autocommit being off with no commit() anywhere, which
    silently discarded every insert at process exit.
    """
    marker = "autocommit-marker-ds"
    try:
        db.register_dataset(marker, Path("a/"))

        # A second, independent connection can only see committed rows.
        with DBCredentials().connect() as other:
            count = 0
            res = other.execute(
                "SELECT count(*) FROM dataset WHERE name = %s;", (marker,)
            ).fetchone()
            if res is not None:
                (count,) = res
        assert count == 1, "the insert was never committed"
    finally:
        db.conn.execute("DELETE FROM dataset WHERE name = %s;", (marker,))


@requires_db
def test_close_is_idempotent() -> None:
    """close() twice is safe."""
    handle = HolodDatabase()
    handle.close()
    handle.close()
    assert handle.conn.closed


@requires_db
def test_context_manager_closes() -> None:
    """Leaving a with-block closes the connection."""
    with HolodDatabase() as handle:
        assert not handle.conn.closed
    assert handle.conn.closed


# -- select mechanics -----------------------------------------------------------
#
# The reads are the mirror of the inserts above: every column named in a SELECT
# has to still exist, the values have to come back as the types the writers put
# in, and the `condition`/`column`/`amount` arguments have to reach the generated
# SQL.


def test_select_dataset(clean_db: HolodDatabase) -> None:
    """A registered dataset comes back whole when filtered on its unique name."""
    marker = "select-marker-ds"
    dataset_id = clean_db.register_dataset(marker, Path("b/"))

    rows = clean_db.select_dataset(condition=("name", marker))

    assert len(rows) == 1
    (row_id, name, root_path, created_at) = rows[0]
    assert (row_id, name, root_path) == (dataset_id, marker, "b")
    assert isinstance(created_at, datetime)


def test_select_recording_session(clean_db: HolodDatabase) -> None:
    """Session optics round-trip as floats, found through their dataset_id."""
    marker_wvl = 0.000405
    marker_lds = 1.89
    marker_pxp = 0.0038
    dataset_id = clean_db.register_dataset("select-rs-ds", Path("c/"))
    session_id = clean_db.insert_recording_session(dataset_id, marker_wvl, marker_lds, marker_pxp)

    rows = clean_db.select_recording_session(condition=("dataset_id", dataset_id))

    assert len(rows) == 1
    assert rows[0] == RecordingRow(session_id, dataset_id, marker_wvl, marker_lds, marker_pxp, None)


def test_select_hologram(clean_db: HolodDatabase) -> None:
    """A hologram round-trips, digest included, filtered on its session."""
    marker_rel_path = Path("d/holo.png")
    marker_z_um = 1500.0
    _, session_id = make_session(clean_db, "select-holo-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, marker_rel_path, marker_z_um, DIGEST)]
    )

    rows = clean_db.select_holograms(condition=("recording_session_id", session_id))

    assert len(rows) == 1
    assert rows[0] == HologramRow(
        holo_id, session_id, marker_rel_path.as_posix(), marker_z_um, DIGEST
    )


def test_select_run(clean_db: HolodDatabase) -> None:
    """A run round-trips, and its JSONB config comes back as a dict, not a string."""
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)

    rows = clean_db.select_run(condition=("id", run_id))

    assert len(rows) == 1
    (row_id, git_commit, config_hash, config, started_at, finished_at, status) = rows[0]
    assert (row_id, git_commit, config_hash) == (run_id, GIT_COMMIT, CONFIG_HASH)
    assert config == CONFIG, "jsonb should load as a dict"
    assert (isinstance(started_at, datetime), finished_at, status) == (True, None, "running")


def test_select_prediction(clean_db: HolodDatabase) -> None:
    """A prediction round-trips with the ids of the run and hologram it points at."""
    marker_epoch = 50
    marker_predicted_z_mm = 0.67
    marker_focus_score = 0.899
    _, session_id = make_session(clean_db, "select-pred-ds")
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("e/holo.png"), 1500.0, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG)
    clean_db.insert_prediction(
        run_id, holo_id, marker_epoch, marker_predicted_z_mm, marker_focus_score
    )

    rows = clean_db.select_prediction(condition=("run_id", run_id))

    assert len(rows) == 1
    assert rows[0] == PredictionRow(
        run_id, holo_id, marker_epoch, marker_predicted_z_mm, marker_focus_score
    )


def test_select_single_column(clean_db: HolodDatabase) -> None:
    """`column` narrows the projection instead of returning the whole row."""
    marker = "select-column-ds"
    clean_db.register_dataset(marker, Path("one/column/"))

    assert clean_db.select_dataset(column="root_path", condition=("name", marker)) == [
        ("one/column",)
    ]


def test_select_amount_limits_rows(clean_db: HolodDatabase) -> None:
    """`amount` caps the result, and 0 is a legal cap rather than "no limit"."""
    _, session_id = make_session(clean_db, "select-amount-ds")
    clean_db.insert_hologram(
        [HologramDetail(session_id, Path(f"img/{i:04d}.png"), 1500.0 + i, DIGEST) for i in range(5)]
    )
    condition = ("recording_session_id", session_id)

    assert len(clean_db.select_holograms(condition=condition, amount=2)) == 2
    assert len(clean_db.select_holograms(condition=condition, amount=99)) == 5
    assert clean_db.select_holograms(condition=condition, amount=0) == []


def test_select_negative_amount_rejected(clean_db: HolodDatabase) -> None:
    """A negative cap is caught in Python; Postgres would raise on the LIMIT anyway."""
    with pytest.raises(ValueError, match="negative"):
        clean_db.select_dataset(amount=-1)


def test_select_no_match_returns_empty(clean_db: HolodDatabase) -> None:
    """A condition matching nothing yields an empty list, not an error."""
    assert clean_db.select_dataset(condition=("name", "no-such-dataset")) == []


def test_select_condition_none_matches_null(clean_db: HolodDatabase) -> None:
    """A None value compares with IS NULL.

    Rendered as `= NULL` it would be neither true nor false but NULL, so the
    query would quietly return nothing at all instead of the unfinished runs.
    """
    moment = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    unfinished = clean_db.insert_run(GIT_COMMIT, CONFIG)
    finished = clean_db.insert_run(GIT_COMMIT, CONFIG, finished_at=moment, status="completed")

    # The cap has to clear however many unfinished runs the database already holds,
    # since this test only asserts about the two rows it wrote itself.
    unfinished_ids = {
        row[0]
        for row in clean_db.select_run(column="id", condition=("finished_at", None), amount=10_000)
        if not isinstance(row, TableRow) and isinstance(row[0], int)
    }
    assert unfinished in unfinished_ids
    assert finished not in unfinished_ids
    assert clean_db.select_run(column="id", condition=("finished_at", moment)) == [(finished,)]


def test_select_quotes_identifiers(clean_db: HolodDatabase) -> None:
    """Column names are quoted, so a hostile one fails as a bad column, not as SQL."""
    with pytest.raises(psycopg.errors.UndefinedColumn), clean_db.transaction():
        clean_db.select_dataset(condition=("name; DROP TABLE dataset", "x"))

    # The table is still there: the statement above never parsed as two statements.
    assert clean_db.select_dataset(column="id", amount=1) is not None


# -- select: row-to-dataclass mapping -------------------------------------------
#
# `_select` has two modes and they return different shapes. With no `column` it
# hands the cursor the table's `class_row` factory and rows arrive as that table's
# dataclass; with a `column` there is no factory and rows arrive as plain tuples.
#
# `class_row` passes the columns to the dataclass **by name**, so a renamed field
# is a TypeError rather than a silently mis-filled row. Field *order* is not free
# either: `__iter__` yields `asdict()` in declaration order, which is what makes
# `(a, b, c) = row` work, and that only lines up with `SELECT *` while the field
# order still matches the table's ordinal order. Both halves are pinned below.


@dataclass(frozen=True)
class Seed:
    """One populated table, plus what it takes to read that row back."""

    select: Callable[..., list[tuple[SQLTypes, ...]] | list[TableRow]]
    condition: tuple[str, Any]
    expected: TableRow


# The dataclass each table must map to. Written out rather than read from
# `get_row_factory()`, so a match arm rewired to the wrong dataclass fails here
# instead of agreeing with itself.
ROW_TYPES: dict[Tables, type[TableRow]] = {
    Tables.Dataset: DatasetRow,
    Tables.Recording_Session: RecordingRow,
    Tables.Hologram: HologramRow,
    Tables.Run: RunRow,
    Tables.Prediction: PredictionRow,
}

SEED_STARTED = datetime(2026, 5, 6, 7, 8, 9, tzinfo=UTC)
SEED_FINISHED = datetime(2026, 5, 6, 9, 8, 7, tzinfo=UTC)


@pytest.fixture
def seeded(clean_db: HolodDatabase) -> dict[Tables, Seed]:
    """Write one row into every table, giving each column a value unlike its neighbours'.

    Distinct values matter: with 0.0 in three float columns a row assembled in the
    wrong order would still compare equal.
    """
    dataset_id = clean_db.register_dataset("mechanics-ds", Path("mech/root"), SEED_STARTED)
    session_id = clean_db.insert_recording_session(dataset_id, 0.000405, 12.5, 0.0038, SEED_STARTED)
    (holo_id,) = clean_db.insert_hologram(
        [HologramDetail(session_id, Path("mech/img/0001.png"), 1234.5, DIGEST)]
    )
    run_id = clean_db.insert_run(GIT_COMMIT, CONFIG, SEED_STARTED, SEED_FINISHED, "completed")
    clean_db.insert_prediction(run_id, holo_id, 7, 2.25, 0.875)

    return {
        Tables.Dataset: Seed(
            clean_db.select_dataset,
            ("id", dataset_id),
            DatasetRow(dataset_id, "mechanics-ds", "mech/root", SEED_STARTED),
        ),
        Tables.Recording_Session: Seed(
            clean_db.select_recording_session,
            ("id", session_id),
            RecordingRow(session_id, dataset_id, 0.000405, 12.5, 0.0038, SEED_STARTED),
        ),
        Tables.Hologram: Seed(
            clean_db.select_holograms,
            ("id", holo_id),
            HologramRow(holo_id, session_id, "mech/img/0001.png", 1234.5, DIGEST),
        ),
        Tables.Run: Seed(
            clean_db.select_run,
            ("id", run_id),
            RunRow(
                run_id, GIT_COMMIT, CONFIG_HASH, CONFIG, SEED_STARTED, SEED_FINISHED, "completed"
            ),
        ),
        Tables.Prediction: Seed(
            clean_db.select_prediction,
            ("run_id", run_id),
            PredictionRow(run_id, holo_id, 7, 2.25, 0.875),
        ),
    }


def ordered_columns(db: HolodDatabase, table: str) -> list[str]:
    """Return the table's column names in the order `SELECT *` emits them."""
    rows = db.conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position;",
        (table,),
    ).fetchall()
    return [str(column) for (column,) in rows]


every_table = pytest.mark.parametrize("table", list(Tables), ids=[t.value for t in Tables])


@every_table
def test_row_factory_defined_for_every_table(table: Tables) -> None:
    """Every table names a row factory.

    `get_row_factory` matches without a fallback arm, so a table added to the enum
    and not to the match returns None. `_select` reads that as "no factory" and
    quietly hands back tuples instead of the dataclass, which is exactly the kind
    of silent downgrade a caller unpacking attributes would not survive.
    """
    assert table.get_row_factory() is not None, f"{table.value} has no row factory"


@every_table
def test_select_returns_that_table_s_dataclass(seeded: dict[Tables, Seed], table: Tables) -> None:
    """A column-less select yields the table's own dataclass, not a tuple or a sibling's."""
    (row,) = seeded[table].select(condition=seeded[table].condition)

    assert type(row) is ROW_TYPES[table]


@every_table
def test_select_preserves_every_field(seeded: dict[Tables, Seed], table: Tables) -> None:
    """Every column comes back carrying the value that was written.

    Dataclass equality compares the type as well as all the fields, so this covers
    "right class" and "nothing mangled in transit" together -- including the ones
    with a shape of their own: `sha256` stays 32 raw bytes rather than a memoryview
    or a hex string, and `config` comes back as the dict that went in, since jsonb
    round-trips through psycopg as an object rather than as JSON text.
    """
    (row,) = seeded[table].select(condition=seeded[table].condition)

    assert row == seeded[table].expected


@every_table
def test_dataclass_fields_match_columns_in_order(
    seeded: dict[Tables, Seed], clean_db: HolodDatabase, table: Tables
) -> None:
    """Field names and order track the table exactly.

    Names are what `class_row` fills the dataclass by; order is what makes the
    `__iter__` unpacking line up with `SELECT *`.
    """
    field_names = [field.name for field in dataclasses.fields(ROW_TYPES[table])]

    assert field_names == ordered_columns(clean_db, table.value)


@every_table
def test_row_unpacks_in_column_order(
    seeded: dict[Tables, Seed], clean_db: HolodDatabase, table: Tables
) -> None:
    """Iterating a row gives the same tuple a raw `SELECT *` would.

    This is the promise `(row_id, name, root_path, created_at) = row` relies on;
    reordering two same-typed fields would swap their values with nothing raising.
    """
    (row,) = seeded[table].select(condition=seeded[table].condition)
    (field, value) = seeded[table].condition
    stmt = sql.SQL("SELECT * FROM {tbl} WHERE {field} = %s;").format(
        tbl=sql.Identifier(table.value), field=sql.Identifier(field)
    )
    raw = clean_db.conn.execute(stmt, (value,)).fetchone()

    assert raw is not None
    assert tuple(row) == tuple(raw)


@every_table
def test_selecting_a_column_gives_tuples(seeded: dict[Tables, Seed], table: Tables) -> None:
    """Naming a column drops the row factory: each row is a 1-tuple holding that value."""
    seed = seeded[table]
    for field in dataclasses.fields(ROW_TYPES[table]):
        rows = seed.select(column=field.name, condition=seed.condition)

        assert type(rows[0]) is tuple, f"{table.value}.{field.name} should not be a dataclass"
        assert rows == [(getattr(seed.expected, field.name),)]


@every_table
def test_selecting_a_column_never_builds_a_dataclass(
    seeded: dict[Tables, Seed], table: Tables
) -> None:
    """The projected value is handed back bare, not wrapped in the table's row type.

    A tuple subclass would satisfy the shape checks above while still being a
    `TableRow`, so the negative is asserted on its own.
    """
    first_field = dataclasses.fields(ROW_TYPES[table])[0].name
    (row,) = seeded[table].select(column=first_field, condition=seeded[table].condition)

    assert not isinstance(row, TableRow)


@every_table
def test_select_unknown_column_rejected(
    seeded: dict[Tables, Seed], clean_db: HolodDatabase, table: Tables
) -> None:
    """A column the table does not have fails loudly rather than returning nothing."""
    with pytest.raises(psycopg.errors.UndefinedColumn), clean_db.transaction():
        seeded[table].select(column="not_a_column", condition=seeded[table].condition)


def test_get_id_type_matches_the_table(clean_db: HolodDatabase) -> None:
    """Each table's id helper returns the value it was handed, and None where the PK is composite.

    The NewTypes erase to int at runtime, so only the value and the None case are
    observable -- but the None case is the one that matters, since `prediction` has
    no `id` column for a caller to ask about.
    """
    for table in Tables:
        expected = None if table is Tables.Prediction else 7
        assert table.get_Id_type(7) == expected, f"{table.value} maps its id wrongly"


def test_select_default_amount_is_fifty(clean_db: HolodDatabase) -> None:
    """The default cap is 50, so an unbounded-looking select cannot pull a whole table."""
    _, session_id = make_session(clean_db, "default-amount-ds")
    clean_db.insert_hologram(
        [
            HologramDetail(session_id, Path(f"img/{i:04d}.png"), 1500.0 + i, DIGEST)
            for i in range(60)
        ]
    )

    assert len(clean_db.select_holograms(condition=("recording_session_id", session_id))) == 50
