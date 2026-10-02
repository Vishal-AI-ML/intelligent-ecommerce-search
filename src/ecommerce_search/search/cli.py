"""Search-index command line: `reindex`, `status`, `embed`, `embed-status` and `model-fetch`.

`reindex` and `status` require an explicit `--database NAME` (there is no silent default to
POSTGRES_DB) and print only that name, never credentials. `reindex` rebuilds missing and stale
documents in one transaction and leaves current documents (and their `built_at`) untouched.
`status` is read-only.

`embed --database NAME` generates missing and stale product embeddings with the pinned local
model (ADR-006), in plan / encode / write phases; any failure writes nothing, and current rows
(and their `embedded_at`) are untouched. `embed-status --database NAME` is read-only and never
loads the model unless `--verify-vectors` is given. Both need the model snapshot only for
encoding and never use the network.

`model-fetch` is the only command that uses the network: it downloads one embedding-model
snapshot, at a full commit hash, into the git-ignored `models/` directory and writes a file
manifest. It needs no database and no settings.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ecommerce_search.config import Settings, get_settings
from ecommerce_search.db.engine import create_db_engine
from ecommerce_search.embeddings.fetch import FetchError, fetch_snapshot
from ecommerce_search.embeddings.provider import Embedder, EmbedderUnavailable
from ecommerce_search.embeddings.spec import SpecError, repository_models_dir
from ecommerce_search.search.dense_indexing import EmbeddingError, embed, embedding_status
from ecommerce_search.search.indexing import IndexingError, index_status, reindex


class CliError(Exception):
    """A user-facing operational error with an already-sanitized message."""


def _settings() -> Settings:
    try:
        return get_settings()
    except ValidationError:
        raise CliError(
            "settings are incomplete or invalid (is POSTGRES_PASSWORD set? is the embedding "
            "model one of the reviewed registry entries?)"
        ) from None


def _engine(database: str) -> Engine:
    return create_db_engine(_settings().model_copy(update={"postgres_db": database}))


def make_embedder(settings: Settings) -> Embedder:
    """The configured local embedding provider (replaced by a fake in tests)."""
    from ecommerce_search.embeddings.sentence_transformers_provider import (
        SentenceTransformerEmbedder,
    )

    models_dir = settings.resolved_models_dir()
    if models_dir is None:
        raise CliError("not running from a source checkout: set EMBEDDING_MODELS_DIR")
    return SentenceTransformerEmbedder(
        settings.embedding_spec(), models_dir, batch_size=settings.embedding_batch_size
    )


UNAVAILABLE_HINT = {
    "snapshot_missing": "the model snapshot is not present; run `model-fetch` (see ADR-006)",
}


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


def cmd_embed(args: argparse.Namespace) -> int:
    print(f"target database: {args.database}")
    settings = _settings()
    embedder = make_embedder(settings)
    spec = settings.embedding_spec()
    print(f"model: {spec.model_id}@{spec.revision}")
    engine = _engine(args.database)
    try:
        result = embed(engine, embedder, rebuild_all=args.all)
    finally:
        engine.dispose()
    print(f"embed: {result.summary()}")
    if result.changed_during_run:
        print(
            f"error: {result.changed_during_run} product(s) changed while embedding and were not "
            "written; run embed again",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_embed_status(args: argparse.Namespace) -> int:
    print(f"target database: {args.database}")
    settings = _settings()
    embedder = make_embedder(settings) if args.verify_vectors else None
    engine = _engine(args.database)
    try:
        with Session(engine) as session:
            status = embedding_status(session, settings.embedding_spec(), embedder)
    finally:
        engine.dispose()
    print(f"embeddings: {status.describe()}")
    return 1 if (args.require_current and not status.current) else 0


def cmd_model_fetch(args: argparse.Namespace) -> int:
    models_dir = args.models_dir or repository_models_dir()
    if models_dir is None:
        raise CliError("not running from a source checkout: pass --models-dir explicitly")
    print(f"models directory: {models_dir.as_posix()}")
    print(f"network: downloading {args.model_id} at revision {args.revision}")
    result = fetch_snapshot(args.model_id, args.revision, models_dir)
    print(f"snapshot: {result.snapshot.as_posix()}")
    print(f"manifest: {result.manifest_path.as_posix()}")
    print(f"files: {len(result.files)} bytes: {result.total_bytes}")
    print(f"declared license (model card front matter): {result.declared_license}")
    return 0


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
    p = sub.add_parser(
        "embed", help="generate missing and stale product embeddings (one write transaction)"
    )
    p.add_argument("--database", required=True, help="REQUIRED: the database that will be written")
    p.add_argument("--all", action="store_true", help="re-embed every product, even current ones")
    p.set_defaults(func=cmd_embed)
    p = sub.add_parser("embed-status", help="read-only: completeness of the product embeddings")
    p.add_argument("--database", required=True, help="REQUIRED: the database that will be read")
    p.add_argument("--require-current", action="store_true", help="exit 1 unless fully current")
    p.add_argument(
        "--verify-vectors",
        action="store_true",
        help="also re-encode current rows with the model and compare the stored vectors",
    )
    p.set_defaults(func=cmd_embed_status)
    p = sub.add_parser(
        "model-fetch",
        help="NETWORK: download an embedding-model snapshot at a pinned commit into models/",
    )
    p.add_argument("--model-id", required=True, help="Hugging Face repository, e.g. org/name")
    p.add_argument("--revision", required=True, help="full 40-character commit hash (immutable)")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="default: the git-ignored models/ directory at the repository root",
    )
    p.set_defaults(func=cmd_model_fetch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (IndexingError, CliError, FetchError, SpecError, EmbeddingError) as exc:
        return _fail(str(exc))
    except EmbedderUnavailable as exc:
        hint = UNAVAILABLE_HINT.get(exc.reason, "details are intentionally not shown")
        return _fail(f"embedding model unavailable ({exc.reason}): {hint}; nothing was written")
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
