"""Explicit model download: the ONLY code in this project that contacts the network.

`fetch_snapshot` downloads exactly the files a sentence-transformers model needs, from the
official Hugging Face repository, at a full immutable commit hash, into the repository's
git-ignored `models/` directory. It then verifies the resolved snapshot path is that exact
revision, refuses repositories that would need remote code, and writes a SHA-256 manifest of
every downloaded file next to the hub folder.

Cache isolation: `huggingface_hub` and `hf_xet` read their cache, token and log locations from
the environment when they are imported. Before importing them, the default downloader points
every such location (`HF_HOME`, `HF_HUB_CACHE`, `HF_ASSETS_CACHE`, `HF_XET_CACHE`,
`HF_XET_LOG_DIR`, `HF_TOKEN_PATH`, `XDG_CACHE_HOME`, ...) inside the selected models directory, so
nothing is read from or written to a user-level or shared cache. If either library was already
imported in this process its locations may already be fixed, so the fetch is refused.

The licence declared in the downloaded model card (README front matter) is reported for the
operator to verify against the official model page; it is not treated as legal advice.
"""

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ecommerce_search.embeddings.spec import (
    check_model_id,
    check_revision,
    repo_folder_name,
    snapshot_dir,
)

# Exact file names (fnmatch patterns without wildcards), so ONNX/OpenVINO/PyTorch-bin variants
# and any Python files are never downloaded.
ALLOW_PATTERNS: tuple[str, ...] = (
    "modules.json",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "model.safetensors",
    "1_Pooling/config.json",
    "2_Normalize/*",
    "README.md",
    "LICENSE",
)
REQUIRED: tuple[str, ...] = ("modules.json", "config.json", "model.safetensors", "README.md")
MANIFEST_VERSION = 1
_FRONT_MATTER_LICENSE = re.compile(r"^license:\s*(?P<value>\S.*?)\s*$", re.MULTILINE)


class FetchError(Exception):
    """The snapshot could not be fetched or failed verification. Messages are sanitized."""


@dataclass(frozen=True)
class FetchResult:
    model_id: str
    revision: str
    snapshot: Path
    manifest_path: Path
    files: list[dict]
    total_bytes: int
    declared_license: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_license(readme: Path) -> str | None:
    """The `license:` value of the model card's YAML front matter, if present."""
    text = readme.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    match = _FRONT_MATTER_LICENSE.search(text[: end if end > 0 else len(text)])
    return match["value"].strip("'\"") if match else None


def file_manifest(snapshot: Path) -> list[dict]:
    """(path, bytes, sha256) of every file in the snapshot (symlinks resolved)."""
    # Sorted by the POSIX path string: Windows path comparison is case-insensitive, so sorting
    # Path objects would give a platform-dependent order.
    files = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    return [
        {"path": name, "bytes": files[name].stat().st_size, "sha256": sha256_file(files[name])}
        for name in sorted(files)
    ]


def verify_snapshot(snapshot: Path) -> None:
    missing = [name for name in REQUIRED if not (snapshot / name).is_file()]
    if missing:
        raise FetchError(f"snapshot is missing required files: {', '.join(missing)}")
    if any(snapshot.rglob("*.py")):
        raise FetchError("snapshot contains Python files; remote code is never used")
    try:
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise FetchError("snapshot config.json is not valid JSON") from None
    if "auto_map" in config:
        raise FetchError("model requires remote code (auto_map); refusing")


# Libraries whose cache/log locations are fixed at import time.
HF_LIBRARIES: tuple[str, ...] = ("huggingface_hub", "hf_xet")


def isolated_hf_environment(models_dir: Path) -> dict[str, str]:
    """Every Hugging Face cache, token and log location, inside `models_dir` (absolute paths)."""
    root = models_dir.resolve()
    home = root / ".hf_home"
    hub = root / "hub"
    values = {
        "HF_HOME": home,
        "HF_HUB_CACHE": hub,
        "HUGGINGFACE_HUB_CACHE": hub,  # legacy name, still read by huggingface_hub
        "HF_ASSETS_CACHE": home / "assets",
        "HUGGINGFACE_ASSETS_CACHE": home / "assets",
        "HF_TOKEN_PATH": home / "token",  # no user token is read: public repositories only
        "HF_XET_CACHE": home / "xet",
        "HF_XET_LOG_DIR": home / "xet" / "logs",
        "XDG_CACHE_HOME": home / "xdg",  # fallback root used when HF_HOME is unset elsewhere
    }
    env = {name: str(path) for name, path in values.items()}
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return env


def _import_snapshot_download() -> Callable[..., str]:
    from huggingface_hub import snapshot_download

    return snapshot_download


def isolated_download(models_dir: Path, **kwargs) -> str:
    """Download with every Hugging Face location confined to `models_dir`."""
    loaded = [name for name in HF_LIBRARIES if name in sys.modules]
    if loaded:
        raise FetchError(
            f"{', '.join(loaded)} is already imported in this process, so its cache locations "
            "may point outside the models directory; run model-fetch in a fresh process"
        )
    os.environ.update(isolated_hf_environment(models_dir))
    snapshot_download = _import_snapshot_download()
    # Sequential downloads: parallel connections were reset by the network in practice, and a
    # few small files plus one weights file gain little from parallelism.
    return snapshot_download(max_workers=1, **kwargs)


def fetch_snapshot(
    model_id: str,
    revision: str,
    models_dir: Path,
    download: Callable[..., str] | None = None,
) -> FetchResult:
    """Fetch one snapshot. `download` replaces the isolated Hugging Face downloader in tests."""
    check_model_id(model_id)
    check_revision(revision)
    expected = snapshot_dir(models_dir, model_id, revision)
    request = {
        "repo_id": model_id,
        "revision": revision,
        "cache_dir": str((models_dir / "hub").resolve()),
        "allow_patterns": list(ALLOW_PATTERNS),
    }
    try:
        if download is None:
            resolved = Path(isolated_download(models_dir, **request))
        else:
            resolved = Path(download(**request))
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"download failed ({type(exc).__name__})") from None
    if resolved.resolve() != expected.resolve():
        raise FetchError("the downloaded snapshot is not the requested revision")
    verify_snapshot(resolved)
    files = file_manifest(resolved)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "model_id": model_id,
        "revision": revision,
        "allow_patterns": list(ALLOW_PATTERNS),
        "declared_license": declared_license(resolved / "README.md"),
        "files": files,
    }
    manifest_dir = models_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{repo_folder_name(model_id)}@{revision}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return FetchResult(
        model_id=model_id,
        revision=revision,
        snapshot=resolved,
        manifest_path=manifest_path,
        files=files,
        total_bytes=sum(f["bytes"] for f in files),
        declared_license=manifest["declared_license"],
    )


def verify_manifest(models_dir: Path, model_id: str, revision: str) -> list[str]:
    """Offline re-check of a fetched snapshot against its manifest. Returns problems found."""
    manifest_path = models_dir / "manifests" / f"{repo_folder_name(model_id)}@{revision}.json"
    snapshot = snapshot_dir(models_dir, model_id, revision)
    if not manifest_path.is_file() or not snapshot.is_dir():
        return ["snapshot or manifest is missing (run model-fetch)"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        return ["manifest does not describe this model and revision"]
    recorded = {f["path"]: f for f in manifest["files"]}
    actual = {f["path"]: f for f in file_manifest(snapshot)}
    problems = [f"missing file {p}" for p in sorted(set(recorded) - set(actual))]
    problems += [f"unexpected file {p}" for p in sorted(set(actual) - set(recorded))]
    problems += [
        f"changed file {p}"
        for p in sorted(set(recorded) & set(actual))
        if recorded[p]["sha256"] != actual[p]["sha256"]
    ]
    return problems
