"""scripts/benchmark_dense.py: pure logic, guards, derivation and cleanup (no DB, no model)."""

import contextlib
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.usefixtures("no_socket_connect")


def _load():
    name = "benchmark_dense_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "benchmark_dense.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()
QUERIES = list(bench.query_set_identity()["normalized"])


# ---- query set ---------------------------------------------------------------------------------


def test_queries_are_the_committed_selection_list():
    selection = sys.modules["embedding_model_selection"]
    assert tuple(selection.QUERIES) == bench.QUERIES
    identity = bench.query_set_identity()
    assert identity["count"] == 12 == len(set(bench.QUERIES))
    assert identity["source"] == "scripts/embedding_model_selection.py QUERIES"
    assert identity == bench.query_set_identity()  # deterministic
    assert len(identity["sha256"]) == len(identity["normalized_sha256"]) == 64


@pytest.mark.parametrize(
    ("queries", "message"),
    [
        (list(bench.QUERIES[:11]), "expected 12"),
        ([*bench.QUERIES[:11], bench.QUERIES[0]], "duplicates"),
        # distinct raw text that normalizes to an existing query ("hp laptop")
        ([*bench.QUERIES[:11], "hp  laptop"], "normalize to the same"),
    ],
)
def test_query_set_identity_refuses_bad_sets(queries, message):
    assert "hp laptop" in bench.QUERIES[:11]
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.query_set_identity(queries)


def test_hashes_are_deterministic_and_sensitive():
    assert bench.vectors_sha256([[0.1, 0.2]]) == bench.vectors_sha256([[0.1, 0.2]])
    assert bench.vectors_sha256([[0.1, 0.2]]) != bench.vectors_sha256([[0.1, 0.20000001]])
    assert bench.ids_sha256(["a", "b"]) != bench.ids_sha256(["b", "a"])


def test_reuses_the_tested_lexical_helpers():
    lexical = sys.modules["benchmark_lexical"]
    assert bench.summarize is lexical.summarize
    assert bench.dumps is lexical.dumps and bench.read_samples is lexical.read_samples
    assert bench.source_fingerprint is lexical.source_fingerprint


# ---- provenance guard --------------------------------------------------------------------------

HEAD = "f" * 40


def fake_git(status=b"", diff=b"", ancestor=True, calls=None):
    def git(root, *args):
        if calls is not None:
            calls.append(args)
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return (HEAD + "\n").encode()
        if args[0] == "merge-base":
            if not ancestor:
                raise bench.GitError("not an ancestor")
            return b""
        if args[0] == "diff":
            return diff
        raise AssertionError(args)

    return git


def test_clean_tree_with_identical_runtime_paths_is_accepted():
    calls = []
    result = bench.check_provenance(ROOT, git=fake_git(calls=calls))
    assert result == {
        "harness_head": HEAD,
        "production_commit": bench.PRODUCTION_COMMIT,
        "guarded": bench.GUARDED_PATHS,
    }
    diff_call = next(c for c in calls if c[0] == "diff")
    assert diff_call[:4] == ("diff", "--name-only", bench.PRODUCTION_COMMIT, HEAD)
    assert diff_call[5:] == bench.GUARDED_PATHS
    assert not any(p.startswith(("scripts", "tests")) for p in diff_call[5:])


def test_guarded_paths_are_exactly_the_approved_set():
    assert bench.GUARDED_PATHS == (
        "src/",
        "migrations/",
        "pyproject.toml",
        "uv.lock",
        "docker-compose.yml",
        "alembic.ini",
    )
    assert bench.PRODUCTION_COMMIT == "0ab972e1cc45dd489b262a39962a4c6debea5c43"


@pytest.mark.parametrize("status", [b" M src/x.py\n", b"?? scripts/new.py\n", b"M  uv.lock\n"])
def test_a_dirty_tree_is_refused(status):
    with pytest.raises(bench.ProvenanceError, match="not clean"):
        bench.check_provenance(ROOT, git=fake_git(status=status))


@pytest.mark.parametrize(
    "changed",
    [
        "src/ecommerce_search/search/dense.py",
        "migrations/versions/0005_x.py",
        "pyproject.toml",
        "uv.lock",
        "docker-compose.yml",
        "alembic.ini",
    ],
)
def test_any_runtime_affecting_difference_is_refused(changed):
    with pytest.raises(bench.ProvenanceError, match="runtime-affecting"):
        bench.check_provenance(ROOT, git=fake_git(diff=f"{changed}\n".encode()))


def test_production_commit_must_be_an_ancestor_of_head():
    with pytest.raises(bench.ProvenanceError, match="not an ancestor"):
        bench.check_provenance(ROOT, git=fake_git(ancestor=False))


class FakeSpec:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    dimension = 384
    max_seq_length = 256
    normalize = True

    def config_sha256(self):
        return "c" * 64


def test_model_manifest_must_verify(tmp_path):
    ok = bench.check_model(tmp_path, FakeSpec(), lambda *a: [])
    assert ok["revision"] == FakeSpec.revision and ok["manifest_problems"] == []
    with pytest.raises(bench.ProvenanceError, match="failed verification"):
        bench.check_model(tmp_path, FakeSpec(), lambda *a: ["changed file model.safetensors"])
    with pytest.raises(bench.ProvenanceError, match="models directory"):
        bench.check_model(None, FakeSpec(), lambda *a: [])


# ---- scratch-database safety -------------------------------------------------------------------


def test_scratch_names_are_unique_and_valid():
    names = {bench.new_scratch_name() for _ in range(50)}
    assert len(names) == 50
    for name in names:
        assert bench.assert_scratch_name(name, "ecommerce_search") == name


@pytest.mark.parametrize(
    "name",
    [
        "ecommerce_search",  # the development database
        "postgres",
        "ecommerce_search_test_0123456789ab",
        "ecommerce_search_bench_0123456789AB",
        "ecommerce_search_bench_0123",
        'ecommerce_search_bench_0123456789ab"; DROP DATABASE x; --',
    ],
)
def test_anything_but_a_bench_name_is_refused(name):
    with pytest.raises(bench.BenchmarkError):
        bench.assert_scratch_name(name, "ecommerce_search")


def test_the_configured_development_database_is_refused_even_if_it_looks_like_a_bench_name():
    name = "ecommerce_search_bench_0123456789ab"
    with pytest.raises(bench.BenchmarkError, match="development database"):
        bench.assert_scratch_name(name, development_db=name)


# ---- phase order -------------------------------------------------------------------------------


def test_phase_schedule_is_exactly_as_declared():
    assert bench.RUNS == 3
    assert [bench.phase_order(r) for r in (1, 2, 3)] == [
        ("C1", "C2"),
        ("C2", "C1"),
        ("C1", "C2"),
    ]
    with pytest.raises(bench.BenchmarkError):
        bench.phase_order(4)


# ---- ranking and recall ------------------------------------------------------------------------


def test_validate_ranking():
    bench.validate_ranking(["A", "B", "C"], [0.1, 0.2, 0.2])
    for ids, distances in (
        (["A", "B"], [0.2, 0.1]),  # not ascending
        (["B", "A"], [0.1, 0.1]),  # tie not by product_id
        (["A", "A"], [0.1, 0.2]),  # duplicate
        (["A"], [0.1, 0.2]),  # mismatched lengths
    ):
        with pytest.raises(bench.BenchmarkError):
            bench.validate_ranking(ids, distances)


def test_strict_and_tie_aware_recall():
    exact_ids, exact_d = ["A", "B", "C"], [0.1, 0.2, 0.3]
    assert bench.strict_recall(exact_ids, ["A", "B", "C"], 3) == 1.0
    assert bench.strict_recall(exact_ids, ["A", "B", "D"], 3) == pytest.approx(2 / 3)
    # D ties with C at the k-th distance: strict recall drops, tie-aware recall does not
    assert bench.tie_aware_recall(exact_d, [0.1, 0.2, 0.3], 3) == 1.0
    assert bench.tie_aware_recall(exact_d, [0.1, 0.2, 0.3 + 1e-12], 3) == 1.0
    assert bench.tie_aware_recall(exact_d, [0.1, 0.2, 0.31], 3) == pytest.approx(2 / 3)
    assert bench.tie_aware_recall(exact_d, [0.1, 0.2], 3) == pytest.approx(2 / 3)  # short list
    assert bench.tie_aware_recall([], [0.1], 1) == 0.0
    assert bench.max_distance_difference({"A": 0.1, "B": 0.2}, {"A": 0.1004, "C": 9}) == (
        pytest.approx(0.0004)
    )


# ---- decision rule -----------------------------------------------------------------------------


def decision(c1=None, c2=None, request=100.0, recalls=(1.0,), natural=(True,)):
    c1 = c1 or {1: 10.0, 2: 10.0, 3: 10.0}
    c2 = c2 or {1: 5.0, 2: 5.0, 3: 5.0}
    return bench.decide(c1, c2, request, list(recalls), list(natural))


def test_all_criteria_met_flags_for_review_and_reports_each_criterion():
    result = decision()
    assert result["flag_hnsw_for_separate_review"] is True
    assert set(result["criteria"]) == {
        "1_latency_every_run",
        "2_end_to_end_share",
        "3_tie_aware_recall",
        "4_natural_plan_uses_hnsw",
    }
    assert all(c["passed"] for c in result["criteria"].values())
    assert result["outcome"].startswith("flag HNSW for a separate review")


def test_relative_saving_boundary_is_inclusive_at_25_percent():
    at = decision(c1={1: 8.0, 2: 8.0, 3: 8.0}, c2={1: 6.0, 2: 6.0, 3: 6.0}, request=40)
    assert at["criteria"]["1_latency_every_run"]["passed"] is True  # exactly 25%, 2 ms
    below = decision(c1={1: 8.0, 2: 8.0, 3: 8.0}, c2={1: 6.01, 2: 6.01, 3: 6.01}, request=1)
    assert below["criteria"]["1_latency_every_run"]["passed"] is False


def test_absolute_saving_boundary_is_inclusive_at_1_ms():
    at = decision(c1={1: 4.0, 2: 4.0, 3: 4.0}, c2={1: 3.0, 2: 3.0, 3: 3.0}, request=20)
    assert at["criteria"]["1_latency_every_run"]["passed"] is True
    below = decision(c1={1: 3.6, 2: 3.6, 3: 3.6}, c2={1: 2.7, 2: 2.7, 3: 2.7}, request=1)
    assert below["criteria"]["1_latency_every_run"]["per_run"][1]["absolute_ok"] is False
    assert below["criteria"]["1_latency_every_run"]["passed"] is False


def test_saving_must_strictly_exceed_the_run_to_run_spread():
    # C1 spread 4 ms; saving of exactly 4 ms in run 1 is not enough
    result = decision(c1={1: 12.0, 2: 16.0, 3: 14.0}, c2={1: 8.0, 2: 6.0, 3: 6.0}, request=1)
    latency = result["criteria"]["1_latency_every_run"]
    assert latency["spread_ms"] == 4.0
    assert latency["per_run"][1]["beats_spread"] is False and latency["passed"] is False


def test_every_run_must_pass():
    result = decision(c1={1: 10.0, 2: 10.0, 3: 10.0}, c2={1: 5.0, 2: 5.0, 3: 9.5}, request=1)
    assert result["criteria"]["1_latency_every_run"]["passed"] is False


def test_end_to_end_share_boundary_is_inclusive_at_5_percent():
    assert decision(request=100.0)["criteria"]["2_end_to_end_share"]["passed"] is True  # 5 of 100
    assert decision(request=100.01)["criteria"]["2_end_to_end_share"]["passed"] is False


def test_recall_must_be_exactly_one_everywhere():
    assert decision(recalls=(1.0, 1.0))["criteria"]["3_tie_aware_recall"]["passed"] is True
    result = decision(recalls=(1.0, 0.98))
    assert result["criteria"]["3_tie_aware_recall"]["passed"] is False
    assert result["flag_hnsw_for_separate_review"] is False
    assert decision(recalls=())["criteria"]["3_tie_aware_recall"]["passed"] is False


def test_natural_plan_must_choose_hnsw_without_forcing():
    result = decision(natural=(True, False))
    assert result["criteria"]["4_natural_plan_uses_hnsw"]["passed"] is False
    assert result["outcome"] == "retain exact scan at 240 rows (approved default)"
    assert decision(natural=())["criteria"]["4_natural_plan_uses_hnsw"]["passed"] is False


# ---- derivation, artifacts and --verify ----------------------------------------------------------

SAMPLES = 2
PRODUCTS = [f"P{i:03d}" for i in range(80)]  # enough for query index 11 at k=50


def make_records(c1_ms=10.0, c2_ms=9.0, request_ms=50.0, natural=False, forced=True):
    records = []
    for run in (1, 2, 3):
        seq = 0

        def emit(record, run=run):
            nonlocal seq
            records.append({"experiment_id": "dense-test", "run": run, "sequence": seq, **record})
            seq += 1

        order = list(bench.phase_order(run))
        emit({"record": "run", "phase_order": order, "pgvector": "0.8.6"})
        emit({"record": "generation", "products": 240, "first_embed_ms": 5000.0})
        emit({"record": "index_check", "indexes": ["pk_product_embeddings"], "ann_indexes": 0})
        emit({"record": "index", "build_ms": 12.0, "size_bytes": 1024})
        for suite in ("B1", "B2", "C1", "C2"):
            ks = [10] if suite == "B1" else [10, 50]
            for k in ks:
                for qi, query in enumerate(QUERIES):
                    ids = PRODUCTS[qi : qi + k]
                    distances = [0.01 * i for i in range(k)]
                    for phase, count in (("cold", 1), ("timed", SAMPLES)):
                        for i in range(count):
                            metrics = {
                                "B1": {
                                    "request_ms": request_ms + i,
                                    "app_total_ms": request_ms - 5 + i,
                                    "query_embedding_ms": 8.0,
                                    "vector_ms": 2.0,
                                },
                                "B2": {"vector_ms": 2.0 + i},
                                "C1": {"ann_sql_ms": c1_ms + 0.0 * i},
                                "C2": {"ann_sql_ms": c2_ms + 0.0 * i},
                            }[suite]
                            emit(
                                {
                                    "record": "sample",
                                    "suite": suite,
                                    "label": bench.SUITES[suite]["label"],
                                    "k": k,
                                    "query": query,
                                    "phase": phase,
                                    "sample_index": i,
                                    "phase_order": order,
                                    "timestamp_utc": f"2026-10-02T10:00:{seq % 60:02d}.000000+00:00",
                                    "result_count": k,
                                    "result_ids_sha256": bench.ids_sha256(ids),
                                    **metrics,
                                }
                            )
                    emit(
                        {
                            "record": "results",
                            "suite": suite,
                            "k": k,
                            "query": query,
                            "ids": ids,
                            "distances": distances,
                        }
                    )
                    if suite != "B1":
                        modes = {
                            "B2": [("B2_production", False)],
                            "C1": [("C1_no_index", False)],
                            "C2": [("C2_forced", forced), ("C2_natural", natural)],
                        }[suite]
                        for mode, uses in modes:
                            emit(
                                {
                                    "record": "plan",
                                    "mode": mode,
                                    "k": k,
                                    "query": query,
                                    "nodes": ["Limit"],
                                    "uses_hnsw": uses,
                                }
                            )
    return records


def make_header():
    return {
        "experiment_id": "dense-test",
        "generated_at_utc": "2026-10-02T10:00:00+00:00",
        "provenance": {
            "harness_head": HEAD,
            "production_commit": bench.PRODUCTION_COMMIT,
            "guarded": list(bench.GUARDED_PATHS),
        },
        "source": {"git_working_tree_dirty": False},
        "model": {"model_id": FakeSpec.model_id, "revision": FakeSpec.revision, "dimension": 384},
        "queries": bench.query_set_identity(),
        "protocol": {"runs": 3, "samples_per_query": SAMPLES, "k_values": [10, 50]},
    }


def test_derivation_reports_retain_exact_when_criteria_fail():
    derived = bench.derive(make_header(), make_records())
    assert derived["decision"]["outcome"] == "retain exact scan at 240 rows (approved default)"
    criteria = derived["decision"]["criteria"]
    assert criteria["3_tie_aware_recall"]["passed"] is True  # identical lists
    assert criteria["4_natural_plan_uses_hnsw"]["passed"] is False
    assert derived["runs"]["2"]["phase_order"] == ["C2", "C1"]
    recall = derived["runs"]["1"]["recall"][f"C2@10|{QUERIES[0]}"]
    assert recall == {
        "strict_recall": 1.0,
        "tie_aware_recall": 1.0,
        "identical_order": True,
        "max_distance_difference": 0.0,
    }
    overhead = derived["runs"]["1"]["suites"]["B1@10"]["overall"]["request_overhead_ms"]
    assert overhead["p50"] == 5.0
    assert "request_overhead_ms = request_ms - app_total_ms" in derived["derived_metric_note"]


def test_derivation_can_flag_when_every_criterion_holds():
    records = make_records(c1_ms=10.0, c2_ms=5.0, request_ms=50.0, natural=True)
    assert bench.derive(make_header(), records)["decision"]["flag_hnsw_for_separate_review"]


def test_production_and_experimental_results_are_labelled_separately():
    assert {s: m["label"] for s, m in bench.SUITES.items()} == {
        "B1": "production",
        "B2": "production",
        "C1": "experimental",
        "C2": "experimental",
    }
    assert all("not production SQL" in bench.SUITES[s]["name"] for s in ("C1", "C2"))
    derived = bench.derive(make_header(), make_records())
    for key, suite in derived["runs"]["1"]["suites"].items():
        assert suite["label"] == ("production" if key[:2] in ("B1", "B2") else "experimental")


def write(tmp_path, records=None, header=None):
    return bench.write_artifacts(
        tmp_path, header or make_header(), records or make_records(), {"power_events": {}}
    )


def test_artifacts_round_trip_and_verify(tmp_path):
    json_path = write(tmp_path)
    assert bench.verify_artifacts(json_path) == []
    raw = json_path.with_name("dense-test.samples.jsonl")
    body = raw.read_bytes()
    assert b"\r" not in body
    write(tmp_path)
    assert raw.read_bytes() == body  # deterministic
    markdown = json_path.with_name("dense-test.md").read_text(encoding="utf-8")
    production, experimental = markdown.split("## EXPERIMENTAL")
    assert "| C1@" not in production and "| C2@" not in production
    assert "not production SQL" in experimental and "| B2@" not in experimental


def test_verify_detects_a_tampered_raw_sample(tmp_path):
    json_path = write(tmp_path)
    raw = json_path.with_name("dense-test.samples.jsonl")
    lines = raw.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if '"ann_sql_ms":10.0' in line)
    lines[index] = lines[index].replace('"ann_sql_ms":10.0', '"ann_sql_ms":1.0')
    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = bench.verify_artifacts(json_path)
    assert "raw samples file hash differs from the summary" in problems
    assert any("does not recompute" in p for p in problems)


def test_verify_detects_a_tampered_decision(tmp_path):
    json_path = write(tmp_path)
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    summary["derived"]["decision"]["flag_hnsw_for_separate_review"] = True
    json_path.write_text(json.dumps(summary), encoding="utf-8")
    assert any("does not recompute" in p for p in bench.verify_artifacts(json_path))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda r: [x for x in r if not (x["record"] == "sample" and x["phase"] == "cold")],
            "cold",
        ),
        (lambda r: [x for x in r if x["record"] != "generation"], "generation"),
        (
            lambda r: [x for x in r if not (x["record"] == "results" and x["suite"] == "C2")],
            "result",
        ),
        (
            lambda r: [x for x in r if not (x["record"] == "plan" and x["mode"] == "C2_natural")],
            "plans",
        ),
    ],
)
def test_incomplete_records_are_refused(mutate, message):
    with pytest.raises(bench.BenchmarkError, match=message):
        bench.derive(make_header(), mutate(make_records()))


def test_inconsistent_records_are_refused():
    records = make_records()
    sample = next(r for r in records if r["record"] == "sample" and r["suite"] == "B2")
    sample["result_ids_sha256"] = "0" * 64
    with pytest.raises(bench.BenchmarkError, match="results changed"):
        bench.derive(make_header(), records)
    records = make_records()
    run = next(r for r in records if r["record"] == "run" and r["run"] == 2)
    run["phase_order"] = ["C1", "C2"]
    with pytest.raises(bench.BenchmarkError, match="phase order"):
        bench.derive(make_header(), records)
    with pytest.raises(bench.BenchmarkError, match="forced C2 plan"):
        bench.derive(make_header(), make_records(forced=False))


def test_bad_result_ordering_is_refused():
    records = make_records()
    result = next(r for r in records if r["record"] == "results" and r["suite"] == "B2")
    result["distances"] = list(reversed(result["distances"]))
    with pytest.raises(bench.BenchmarkError, match="ascending distance"):
        bench.derive(make_header(), records)


# ---- cleanup through fakes ---------------------------------------------------------------------


class FakeConn:
    def __init__(self, log, fail_on=None):
        self.log, self.fail_on = log, fail_on

    def execute(self, statement, params=None):
        sql = str(statement)
        self.log.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("database failure")
        return self

    def scalar_one(self):
        return 4096


class FakeEngine:
    def __init__(self, fail_on=None):
        self.log, self.fail_on = [], fail_on

    @contextlib.contextmanager
    def connect(self):
        yield FakeConn(self.log, self.fail_on)

    begin = connect


def test_scratch_database_is_dropped_even_when_the_run_fails():
    admin = FakeEngine()
    name = bench.new_scratch_name()
    with (
        pytest.raises(RuntimeError, match="boom"),
        bench.scratch_database(admin, name, "ecommerce_search"),
    ):
        raise RuntimeError("boom")
    assert admin.log == [
        f'CREATE DATABASE "{name}"',
        f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)',
    ]


def test_scratch_database_refuses_the_development_database_before_any_sql():
    admin = FakeEngine()
    with (
        pytest.raises(bench.BenchmarkError),
        bench.scratch_database(admin, "ecommerce_search", "ecommerce_search"),
    ):
        pass
    assert admin.log == []


def test_temporary_index_is_dropped_even_when_measurement_fails():
    engine = FakeEngine()
    with (
        pytest.raises(RuntimeError, match="boom"),
        bench.temporary_hnsw_index(engine, 16, 64) as built,
    ):
        assert built["size_bytes"] == 4096 and built["build_ms"] >= 0
        raise RuntimeError("boom")
    assert engine.log[0].startswith(f"CREATE INDEX {bench.INDEX_NAME} ON product_embeddings")
    assert "vector_cosine_ops" in engine.log[0] and "m = 16, ef_construction = 64" in engine.log[0]
    assert engine.log[-1] == f"DROP INDEX IF EXISTS {bench.INDEX_NAME}"


@pytest.mark.parametrize(("m", "efc"), [(1, 64), (16, 2), ("16", 64)])
def test_invalid_hnsw_parameters_execute_nothing(m, efc):
    engine = FakeEngine()
    with pytest.raises(bench.BenchmarkError), bench.temporary_hnsw_index(engine, m, efc):
        pass
    assert engine.log == []


# ---- network guard, plans, wrapper and CLI ---------------------------------------------------------


def test_network_guard_refuses_non_loopback(monkeypatch):
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: calls.append(host) or [])
    monkeypatch.setattr(socket, "create_connection", lambda address, *a, **k: calls.append(address))
    bench.install_network_guard()
    with pytest.raises(OSError, match="network access is disabled"):
        socket.getaddrinfo("huggingface.co", 443)
    with pytest.raises(OSError, match="network access is disabled"):
        socket.create_connection(("example.com", 443))
    socket.getaddrinfo("127.0.0.1", 5432)
    socket.create_connection(("localhost", 5432))
    assert calls == ["127.0.0.1", ("localhost", 5432)]


def test_plan_nodes_detects_the_hnsw_index():
    plan = [
        {
            "Plan": {
                "Node Type": "Limit",
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Index Name": bench.INDEX_NAME,
                        "Relation Name": "product_embeddings",
                    }
                ],
            }
        }
    ]
    nodes, uses = bench.plan_nodes(plan)
    assert uses is True and nodes[1].startswith("Index Scan [")
    seq = [{"Plan": {"Node Type": "Seq Scan", "Relation Name": "product_embeddings"}}]
    assert bench.plan_nodes(json.dumps(seq)) == (["Seq Scan on product_embeddings"], False)


def test_encode_timer_is_a_pure_pass_through():
    class Inner:
        spec = "spec"

        def __init__(self):
            self.calls = []

        def load(self):
            return 1.5

        def count_tokens(self, texts, kind):
            self.calls.append(("count", kind))
            return [3]

        def embed_documents(self, texts):
            self.calls.append(("docs", tuple(texts)))
            return [[1.0]]

        def embed_query(self, text):
            return [2.0]

    inner = Inner()
    timer = bench.EncodeTimer(inner)
    assert timer.spec == "spec" and timer.load() == 1.5
    assert timer.count_tokens(["a"], "document") == [3]
    assert timer.embed_documents(["a"]) == [[1.0]] and timer.embed_query("q") == [2.0]
    assert inner.calls == [("count", "document"), ("docs", ("a",))]
    assert timer.encode_ms >= 0


@pytest.mark.parametrize(
    "argv", [["--hnsw-ef-search", "40"], ["--samples", "0"], ["--warmup", "-1"]]
)
def test_cli_rejects_settings_that_would_truncate_or_skip_samples(argv):
    with pytest.raises(SystemExit):
        bench.main(argv)


def test_importing_the_harness_loads_no_model_library():
    code = (
        "import importlib.util, sys; "
        "spec = importlib.util.spec_from_file_location('b', 'scripts/benchmark_dense.py'); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "print(sorted({'torch', 'sentence_transformers', 'psutil'} & set(sys.modules)))"
    )
    done = subprocess.run(  # noqa: S603 - fixed argv
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "[]"


@pytest.mark.parametrize("phase", ["cold", "timed"])
def test_every_raw_sample_is_covered_by_recomputation(tmp_path, phase):
    json_path = write(tmp_path)
    raw = json_path.with_name("dense-test.samples.jsonl")
    lines = raw.read_text(encoding="utf-8").splitlines()
    index = next(
        i for i, line in enumerate(lines) if f'"phase":"{phase}"' in line and '"suite":"C1"' in line
    )
    record = json.loads(lines[index])
    record["ann_sql_ms"] = 123.456
    lines[index] = bench.dumps(record)
    body = "\n".join(lines) + "\n"
    raw.write_text(body, encoding="utf-8")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    summary["raw_samples"]["sha256"] = __import__("hashlib").sha256(body.encode()).hexdigest()
    json_path.write_text(
        json.dumps(summary), encoding="utf-8"
    )  # hash forged: recompute must catch it
    assert any("does not recompute" in p for p in bench.verify_artifacts(json_path))
