"""The benchmark's deterministic source fingerprint (uses throwaway Git repositories)."""

import subprocess
from pathlib import Path

import pytest
from benchmark_support import load_benchmark

bench = load_benchmark()

GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid"]


def git(repo: Path, *args: str) -> None:
    subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True)  # noqa: S603


def write(repo: Path, relative: str, content: str, newline: str = "\n") -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.replace("\n", newline).encode("utf-8"))


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    write(root, ".gitattributes", "* text=auto eol=lf\n")
    write(root, ".gitignore", ".env\ndata/processed/*\n__pycache__/\n")
    write(root, "src/pkg/module.py", "VALUE = 1\nOTHER = 2\n")
    write(root, "migrations/versions/0001.py", "revision = '0001'\n")
    write(root, "scripts/tool.py", "print('tool')\n")
    write(root, "pyproject.toml", "[project]\nname = 'x'\n")
    write(root, "docs/notes.md", "documentation\n")
    write(root, "tests/test_x.py", "def test_x():\n    pass\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def fingerprint(repo: Path) -> dict:
    return bench.source_fingerprint(repo)


def test_unchanged_source_gives_the_same_fingerprint(repo):
    first, second = fingerprint(repo), fingerprint(repo)
    assert first == second
    assert first["source_tree_sha256"] == second["source_tree_sha256"]
    assert first["git_working_tree_dirty"] is False and first["untracked_files"] == {}
    assert len(first["base_git_commit"]) == 40 and len(first["source_tree_sha256"]) == 64


def test_changing_a_tracked_file_changes_the_fingerprint_and_the_diff_hash(repo):
    before = fingerprint(repo)
    write(repo, "src/pkg/module.py", "VALUE = 2\nOTHER = 2\n")
    changed = fingerprint(repo)
    assert changed["source_tree_sha256"] != before["source_tree_sha256"]
    assert changed["git_diff_sha256"] != before["git_diff_sha256"]
    assert changed["git_working_tree_dirty"] is True
    assert changed["base_git_commit"] == before["base_git_commit"]
    write(repo, "src/pkg/module.py", "VALUE = 1\nOTHER = 2\n")  # reverting restores it exactly
    assert fingerprint(repo) == before


def test_changing_an_untracked_m3_source_file_changes_the_fingerprint(repo):
    before = fingerprint(repo)
    write(repo, "src/pkg/search_new.py", "A = 1\n")
    added = fingerprint(repo)
    assert added["source_tree_sha256"] != before["source_tree_sha256"]
    assert set(added["untracked_files"]) == {"src/pkg/search_new.py"}
    assert added["git_working_tree_dirty"] is True
    write(repo, "src/pkg/search_new.py", "A = 2\n")
    edited = fingerprint(repo)
    assert edited["source_tree_sha256"] not in (
        before["source_tree_sha256"],
        added["source_tree_sha256"],
    )
    assert edited["untracked_files"] != added["untracked_files"]


def test_ignored_files_and_benchmark_outputs_do_not_affect_it(repo):
    before = fingerprint(repo)
    write(repo, ".env", "POSTGRES_PASSWORD=never-fingerprint-me\n")
    write(repo, "data/processed/search_benchmark/run.json", "{}\n")
    write(repo, "src/pkg/__pycache__/module.cpython-312.pyc", "binary-ish\n")
    write(repo, ".claude/settings.local.json", "{}\n")  # not ignored here, still excluded
    write(repo, "docs/notes.md", "documentation changed\n")  # docs cannot change what runs
    write(repo, "tests/test_x.py", "def test_x():\n    assert True\n")
    after = fingerprint(repo)
    assert after == before


def test_no_secret_or_ignored_path_is_ever_fingerprinted(repo):
    write(repo, ".env", "POSTGRES_PASSWORD=secret\n")
    write(repo, ".env.local", "X=1\n")
    write(repo, ".claude/x.json", "{}\n")
    write(repo, "data/processed/a.json", "{}\n")
    git(repo, "add", "-f", ".env", ".env.local", ".claude/x.json", "data/processed/a.json")
    git(repo, "commit", "-q", "-m", "force-added local files")  # even tracked, they are excluded
    result = fingerprint(repo)
    paths = set(result["source_files"])
    assert paths == {
        ".gitattributes",
        "migrations/versions/0001.py",
        "pyproject.toml",
        "scripts/tool.py",
        "src/pkg/module.py",
    }
    for forbidden in (".env", ".env.local", ".claude/x.json", "data/processed/a.json"):
        assert forbidden not in paths
    assert "POSTGRES_PASSWORD" not in str(result)


def test_env_example_is_a_relevant_config_file(repo):
    before = fingerprint(repo)
    write(repo, ".env.example", "POSTGRES_PASSWORD=\n")
    assert fingerprint(repo)["source_tree_sha256"] != before["source_tree_sha256"]


def test_line_ending_differences_do_not_change_the_fingerprint(repo):
    before = fingerprint(repo)
    write(repo, "src/pkg/module.py", "VALUE = 1\nOTHER = 2\n", newline="\r\n")
    after = fingerprint(repo)
    assert after["source_tree_sha256"] == before["source_tree_sha256"]
    assert after["git_diff_sha256"] == before["git_diff_sha256"]


def test_deleted_tracked_files_are_recorded_as_deleted(repo):
    (repo / "scripts" / "tool.py").unlink()
    result = fingerprint(repo)
    assert result["source_files"]["scripts/tool.py"] == "deleted"
    assert result["git_working_tree_dirty"] is True


def test_missing_git_fails_clearly(repo, monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(bench.subprocess, "run", no_git)
    with pytest.raises(bench.ProvenanceError, match="git is unavailable"):
        fingerprint(repo)


def test_a_directory_that_is_not_a_repository_fails_clearly(tmp_path):
    with pytest.raises(bench.ProvenanceError, match="failed"):
        fingerprint(tmp_path)


def test_experiment_ids_are_unique_and_well_formed():
    ids = {bench.new_experiment_id() for _ in range(5)}
    assert len(ids) == 5 and all(i.startswith("lexical-v0-") for i in ids)
