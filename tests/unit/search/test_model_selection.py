"""Pure helpers of scripts/embedding_model_selection.py (no model, no network)."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
REV = "0123456789abcdef0123456789abcdef01234567"

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def _load():
    name = "embedding_model_selection_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "embedding_model_selection.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sel = _load()


def test_model_libraries_are_imported_only_inside_the_child():
    source = (ROOT / "scripts" / "embedding_model_selection.py").read_text(encoding="utf-8")
    header = source.split("def child_main")[0]
    assert "import torch" not in header and "import sentence_transformers" not in header


def test_parse_candidate():
    assert sel.parse_candidate(f"org/name@{REV}") == ("org/name", REV)
    for bad in ("org/name", "org/name@main", "name@" + REV):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            sel.parse_candidate(bad)


def test_shortlist_is_exactly_the_approved_three():
    assert set(sel.CANDIDATE_PREFIXES) == {
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-small-v2",
    }


def test_top_k_is_exact_cosine_with_product_id_ties():
    docs = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]]
    ids = ["B", "C", "A", "D"]
    hits = sel.top_k([1.0, 0.0], docs, ids, 3)
    assert [h["product_id"] for h in hits] == ["A", "B", "D"]
    assert hits[0]["score"] == hits[1]["score"] == 1.0


def test_overlap_at_k():
    assert sel.overlap_at_k(["a", "b"], ["b", "c"]) == 0.5
    assert sel.overlap_at_k([], []) == 1.0


def model(p50_runs, rss_runs, size, passed=True):
    return {
        "gates": {"all_passed": passed},
        "snapshot_bytes": size,
        "ranking": {
            "query_encode_p50_ms": sel.per_run_criterion(dict(enumerate(p50_runs, 1))),
            "peak_rss_mb": sel.per_run_criterion(dict(enumerate(rss_runs, 1))),
        },
    }


def test_rule_picks_clearly_faster_model():
    decision = sel.decide(
        {"a": model([5.0, 5.1, 5.2], [500, 500, 500], 90), "b": model([9, 9, 9], [400] * 3, 80)}
    )
    assert decision["winner"] == "a"
    assert decision["trace"][0].startswith("query_encode_p50_ms: a wins")


def test_rule_falls_through_on_ties():
    decision = sel.decide(
        {
            "a": model([5.0, 6.0, 5.5], [500, 501, 500], 90),
            "b": model([5.2, 5.3, 5.4], [400, 400, 401], 80),
        }
    )
    assert decision["winner"] == "b"
    assert "tie" in decision["trace"][0] and decision["trace"][1].startswith("peak_rss_mb: b")


def test_rule_ignores_models_that_fail_a_gate():
    decision = sel.decide(
        {"fast": model([1, 1, 1], [1, 1, 1], 1, passed=False), "slow": model([9] * 3, [9] * 3, 9)}
    )
    assert decision["winner"] == "slow" and decision["passing"] == ["slow"]
    assert sel.decide({"x": model([1] * 3, [1] * 3, 1, passed=False)})["winner"] is None


def test_snapshot_spec_reads_dimension_and_length_from_the_snapshot(tmp_path):
    snap = tmp_path / "hub" / "models--intfloat--e5-small-v2" / "snapshots" / REV
    (snap / "1_Pooling").mkdir(parents=True)
    (snap / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 512}))
    (snap / "1_Pooling" / "config.json").write_text(json.dumps({"word_embedding_dimension": 384}))
    spec = sel.snapshot_spec(tmp_path, "intfloat/e5-small-v2", REV)
    assert (spec.dimension, spec.max_seq_length) == (384, 512)
    assert (spec.query_prefix, spec.document_prefix) == ("query: ", "passage: ")
    assert spec.normalize is True


def test_experiment_candidate_config_matches_the_production_registry():
    from ecommerce_search.embeddings.spec import EMBEDDING_MODELS

    for model_id, spec in EMBEDDING_MODELS.items():
        prefixes = sel.CANDIDATE_PREFIXES[model_id]
        assert (prefixes["query_prefix"], prefixes["document_prefix"]) == (
            spec.query_prefix,
            spec.document_prefix,
        )
