"""Search-index command line: `reindex` and `status`.

Both require an explicit `--database NAME` (there is no silent default to POSTGRES_DB) and print
only that name, never credentials. `reindex` rebuilds missing and stale documents in one
transaction and leaves current documents (and their `built_at`) untouched. `status` is read-only.
"""

import argparse
import sys
from collections.abc import Sequence

import psycopg
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.config import get_settings
from ecommerce_search.db.engine import create_db_engine
from ecommerce_search.search.indexing import IndexingError, index_status, reindex


class CliError(Exception):
    """A user-facing operational error with an already-sanitized message."""


def _engine(database: str) -> Engine:
    try:
        settings = get_settings()
    except ValidationError:
        raise CliError(
            "database settings are incomplete or invalid (is POSTGRES_PASSWORD set?)"
        ) from None
    return create_db_engine(settings.model_copy(update={"postgres_db": database}))


def cmd_reindex(args: argparse.Namespace) -> int:
    print(f"target database: {args.database}")
    engine = _engine(args.database)
    try:
        with Session(engine) as session, session.begin():
            result = reindex(session)
    finally:
        engine.dispose()
    print(f"reindex: {result.summary()}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(f"target database: {args.database}")
    engine = _engine(args.database)
    try:
        with Session(engine) as session:
            status = index_status(session)
    finally:
        engine.dispose()
    print(f"search index: {status.describe()}")
    return 1 if (args.require_current and not status.current) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ecommerce_search.search", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reindex", help="build missing and stale search documents (one transaction)")
    p.add_argument("--database", required=True, help="REQUIRED: the database that will be written")
    p.set_defaults(func=cmd_reindex)
    p = sub.add_parser("status", help="read-only: completeness of the search index")
    p.add_argument("--database", required=True, help="REQUIRED: the database that will be read")
    p.add_argument("--require-current", action="store_true", help="exit 1 unless fully current")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (IndexingError, CliError) as exc:
        return _fail(str(exc))
    except (SQLAlchemyError, psycopg.Error) as exc:
        return _fail(
            f"database error ({type(exc).__name__}); check that the database exists, is reachable "
            "and has been migrated to head (`uv run alembic upgrade head`). Details are "
            "intentionally not shown"
        )


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
