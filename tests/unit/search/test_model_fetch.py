"""Model fetching with the downloader replaced by a local fake (no network is ever used)."""

import json
import os
import sys
import types
from pathlib import Path

import pytest

from ecommerce_search.embeddings import fetch
from ecommerce_search.embeddings.fetch import FetchError, fetch_snapshot, verify_manifest
from ecommerce_search.embeddings.spec import SpecError, repository_models_dir, snapshot_dir
from ecommerce_search.search.cli import main

pytestmark = pytest.mark.usefixtures("no_socket_connect")

MODEL = "org/fake"
REV = "0123456789abcdef0123456789abcdef01234567"
README = "---\nlanguage: en\nlicense: apache-2.0\ntags:\n- x\n---\n# Fake\n"


def fake_download(files: dict[str, str] | None = None, *, resolved_rev: str = REV):
    calls: list[dict] = []
    files = files or {
        "modules.json": "[]",
        "config.json": json.dumps({"hidden_size": 4}),
        "model.safetensors": "weights",
        "README.md": README,
        "1_Pooling/config.json": "{}",
    }

    def download(**kwargs):
        calls.append(kwargs)
        cache = Path(kwargs["cache_dir"])
        target = cache / "models--org--fake" / "snapshots" / resolved_rev
        for name, content in files.items():
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).write_text(content, encoding="utf-8")
        return str(target)

    return download, calls


def test_fetch_downloads_exact_files_at_the_pinned_revision(tmp_path):
    download, calls = fake_download()
    result = fetch_snapshot(MODEL, REV, tmp_path, download=download)
    (call,) = calls
    assert call["repo_id"] == MODEL and call["revision"] == REV
    assert Path(call["cache_dir"]) == tmp_path / "hub"  # never HF_HOME or a shared cache
    assert all("*" not in p or p == "2_Normalize/*" for p in call["allow_patterns"])
    assert not any(p.endswith((".py", ".bin", ".onnx")) for p in call["allow_patterns"])
    assert result.snapshot == snapshot_dir(tmp_path, MODEL, REV)
    assert result.declared_license == "apache-2.0"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [f["path"] for f in manifest["files"]] == [
        "1_Pooling/config.json",
        "README.md",
        "config.json",
        "model.safetensors",
        "modules.json",
    ]
    assert manifest["revision"] == REV and manifest["model_id"] == MODEL
    assert verify_manifest(tmp_path, MODEL, REV) == []


def test_a_changed_file_is_detected_offline(tmp_path):
    download, _ = fake_download()
    result = fetch_snapshot(MODEL, REV, tmp_path, download=download)
    (result.snapshot / "model.safetensors").write_text("tampered", encoding="utf-8")
    (result.snapshot / "extra.txt").write_text("x", encoding="utf-8")
    assert verify_manifest(tmp_path, MODEL, REV) == [
        "unexpected file extra.txt",
        "changed file model.safetensors",
    ]
    assert verify_manifest(tmp_path, MODEL, "f" * 40) == [
        "snapshot or manifest is missing (run model-fetch)"
    ]


@pytest.mark.parametrize("revision", ["main", "v1", REV[:12]])
def test_mutable_revisions_are_refused_before_any_download(tmp_path, revision):
    download, calls = fake_download()
    with pytest.raises(SpecError):
        fetch_snapshot(MODEL, revision, tmp_path, download=download)
    assert calls == []


def test_a_different_resolved_revision_is_refused(tmp_path):
    download, _ = fake_download(resolved_rev="f" * 40)
    with pytest.raises(FetchError, match="not the requested revision"):
        fetch_snapshot(MODEL, REV, tmp_path, download=download)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"modules.json": "[]", "config.json": "{}", "README.md": README}, "missing required"),
        (
            {
                "modules.json": "[]",
                "config.json": json.dumps({"auto_map": {"AutoModel": "x.Y"}}),
                "model.safetensors": "w",
                "README.md": README,
            },
            "remote code",
        ),
        (
            {
                "modules.json": "[]",
                "config.json": "{}",
                "model.safetensors": "w",
                "README.md": README,
                "modeling.py": "print(1)",
            },
            "Python files",
        ),
    ],
)
def test_unsafe_or_incomplete_snapshots_are_refused(tmp_path, files, message):
    download, _ = fake_download(files)
    with pytest.raises(FetchError, match=message):
        fetch_snapshot(MODEL, REV, tmp_path, download=download)


def test_download_errors_are_sanitized(tmp_path):
    def broken(**kwargs):
        raise ConnectionError("https://secret.example/token=abc")

    with pytest.raises(FetchError) as info:
        fetch_snapshot(MODEL, REV, tmp_path, download=broken)
    assert str(info.value) == "download failed (ConnectionError)"


def test_declared_license_parsing(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("---\nlicense: 'mit'\n---\n", encoding="utf-8")
    assert fetch.declared_license(readme) == "mit"
    readme.write_text("# no front matter\nlicense: mit\n", encoding="utf-8")
    assert fetch.declared_license(readme) is None


def test_cli_model_fetch(tmp_path, monkeypatch, capsys):
    download, _ = fake_download()
    monkeypatch.setattr(
        "ecommerce_search.search.cli.fetch_snapshot",
        lambda m, r, d: fetch_snapshot(m, r, d, download=download),
    )
    code = main(
        ["model-fetch", "--model-id", MODEL, "--revision", REV, "--models-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"models directory: {tmp_path.as_posix()}" in out
    assert "network: downloading org/fake" in out and "declared license" in out


def test_cli_model_fetch_refuses_a_branch_name(tmp_path, capsys):
    code = main(
        ["model-fetch", "--model-id", MODEL, "--revision", "main", "--models-dir", str(tmp_path)]
    )
    assert code == 1
    assert "40-character" in capsys.readouterr().err


# ---- cache isolation -------------------------------------------------------------------------

HF_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_ASSETS_CACHE",
    "HUGGINGFACE_ASSETS_CACHE",
    "HF_TOKEN_PATH",
    "HF_XET_CACHE",
    "HF_XET_LOG_DIR",
    "XDG_CACHE_HOME",
)


@pytest.fixture
def outside_home(tmp_path, monkeypatch):
    """A user-level HF_HOME outside the models tree; every touched variable is restored."""
    outside = tmp_path / "user-hf-home"
    for key in (*HF_ENV_KEYS, "HF_HUB_DISABLE_TELEMETRY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HOME", str(outside))
    for name in fetch.HF_LIBRARIES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return outside


def test_isolated_environment_points_every_location_inside_the_models_dir(tmp_path):
    models = tmp_path / "models"
    env = fetch.isolated_hf_environment(models)
    assert set(HF_ENV_KEYS) <= set(env)
    for key in HF_ENV_KEYS:
        assert Path(env[key]).is_absolute()
        assert Path(env[key]).is_relative_to(models.resolve()), key
    assert Path(env["HF_HUB_CACHE"]) == (models / "hub").resolve()
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"


@pytest.mark.parametrize("library", ["huggingface_hub", "hf_xet"])
def test_fetch_is_refused_if_a_hugging_face_library_is_already_imported(
    tmp_path, monkeypatch, outside_home, library
):
    monkeypatch.setitem(sys.modules, library, types.ModuleType(library))
    imported = []
    monkeypatch.setattr(fetch, "_import_snapshot_download", lambda: imported.append(1))
    with pytest.raises(FetchError, match="already imported"):
        fetch_snapshot(MODEL, REV, tmp_path / "models")
    assert imported == []  # refused before any import or download
    assert os.environ["HF_HOME"] == str(outside_home)  # environment untouched


def test_environment_is_set_before_the_import_and_nothing_is_written_outside(
    tmp_path, monkeypatch, outside_home
):
    models = tmp_path / "models"
    seen = {}

    def fake_import():
        # what huggingface_hub/hf_xet would read when imported
        seen.update({key: os.environ.get(key) for key in HF_ENV_KEYS})
        download, _ = fake_download()

        def snapshot_download(**kwargs):
            seen["max_workers"] = kwargs.pop("max_workers")
            for key in ("HF_XET_LOG_DIR", "HF_XET_CACHE", "HF_ASSETS_CACHE"):
                target = Path(os.environ[key])
                target.mkdir(parents=True, exist_ok=True)
                (target / "written.log").write_text("x", encoding="utf-8")
            return download(**kwargs)

        return snapshot_download

    monkeypatch.setattr(fetch, "_import_snapshot_download", fake_import)
    result = fetch_snapshot(MODEL, REV, models)
    assert result.snapshot == snapshot_dir(models, MODEL, REV)
    assert seen["max_workers"] == 1
    for key in HF_ENV_KEYS:
        assert Path(seen[key]).is_relative_to(models.resolve()), key
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written and all(p.is_relative_to(models) for p in written)
    assert not outside_home.exists()


# ---- default models directory ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]


def test_default_models_dir_is_the_repository_root_from_any_cwd(tmp_path, monkeypatch, capsys):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert repository_models_dir() == ROOT / "models"
    received = {}

    def fake_fetch(model_id, revision, models_dir):
        received["models_dir"] = models_dir
        raise FetchError("stop here (test)")

    monkeypatch.setattr("ecommerce_search.search.cli.fetch_snapshot", fake_fetch)
    assert main(["model-fetch", "--model-id", MODEL, "--revision", REV]) == 1
    assert received["models_dir"] == ROOT / "models"
    assert not (nested / "models").exists()


def test_explicit_models_dir_overrides_the_default(tmp_path, monkeypatch):
    received = {}

    def fake_fetch(model_id, revision, models_dir):
        received["models_dir"] = models_dir
        raise FetchError("stop here (test)")

    monkeypatch.setattr("ecommerce_search.search.cli.fetch_snapshot", fake_fetch)
    main(["model-fetch", "--model-id", MODEL, "--revision", REV, "--models-dir", str(tmp_path)])
    assert received["models_dir"] == tmp_path


def test_the_default_is_git_ignored():
    import subprocess

    done = subprocess.run(  # noqa: S603 - fixed argv
        ["git", "check-ignore", "-q", "models/hub/x"],  # noqa: S607
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0


def test_no_default_outside_a_source_checkout(monkeypatch, capsys):
    monkeypatch.setattr("ecommerce_search.search.cli.repository_models_dir", lambda: None)
    assert main(["model-fetch", "--model-id", MODEL, "--revision", REV]) == 1
    assert "pass --models-dir" in capsys.readouterr().err
