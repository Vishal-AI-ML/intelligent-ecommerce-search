"""Raw benchmark samples: completeness, indexing, abort rules and exact recomputation."""

import json
import math
import time
from datetime import datetime

import pytest
from benchmark_support import load_benchmark

bench = load_benchmark()

EXPERIMENT = "lexical-v0-test-0001"
SOURCE = {
    "base_git_commit": "a" * 40,
    "git_working_tree_dirty": True,
    "git_diff_sha256": "b" * 64,
    "untracked_files": {},
    "source_files": {},
    "source_file_count": 0,
    "source_tree_sha256": "c" * 64,
}


def payload(query: str, count: int = 0, lexical: float = 1.0, total: float = 1.1) -> dict:
    return {
        "query": query,
        "search_version": "v0_lexical",
        "document_version": "1",
        "top_k": 10,
        "result_count": count,
        "results": [],
        "tsquery": "'x'",
        "applied_filters": [],
        "latency_ms": {"lexical_ms": lexical, "total_ms": total},
    }


class FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeClient:
    """Stands in for TestClient: counts calls and can fail or stall a chosen call."""

    def __init__(self, fail_on=None, status=500, stall_on=None, stall=0.0, counts=None):
        self.calls = 0
        self.fail_on, self.status = fail_on, status
        self.stall_on, self.stall = stall_on, stall
        self.counts = counts or {}

    def get(self, url, params):
        self.calls += 1
        if self.stall_on == self.calls:
            time.sleep(self.stall)
        if self.fail_on == self.calls:
            return FakeResponse(self.status, {"detail": "boom"})
        query = params["q"]
        count = self.counts.get(query, 0)
        return FakeResponse(200, payload(query, count, lexical=0.5 + (self.calls % 7) / 10))


def collect(client, run=1, queries=("q0", "q1", "q2"), warmup=2, samples=7):
    return bench.collect_run(
        client,
        experiment_id=EXPERIMENT,
        run=run,
        queries=list(queries),
        warmup=warmup,
        samples=samples,
    )


# ---- completeness and indexing -----------------------------------------------------------------


def test_expected_raw_sample_count_and_untimed_warmups():
    client = FakeClient()
    records = collect(client)
    timed = [r for r in records if r["phase"] == "timed"]
    cold = [r for r in records if r["phase"] == "cold"]
    assert (len(records), len(timed), len(cold)) == (24, 21, 3)  # 3 x (1 cold + 7 timed)
    assert client.calls == 3 * (1 + 2 + 7)  # warm-ups run but are not recorded


def test_every_sample_has_the_required_fields_and_indexing():
    records = collect(FakeClient(), run=2)
    required = {
        "record",
        "experiment_id",
        "run",
        "query_index",
        "query",
        "phase",
        "sample_index",
        "sequence",
        "timestamp_utc",
        "status_code",
        "result_count",
        "lexical_ms",
        "app_total_ms",
        "serialization_ms",
        "request_ms",
        "outlier",
        "outlier_metrics",
    }
    for record in records:
        assert required <= set(record)
        assert record["experiment_id"] == EXPERIMENT and record["run"] == 2
        assert record["status_code"] == 200 and record["record"] == "sample"
        assert datetime.fromisoformat(record["timestamp_utc"]).utcoffset().total_seconds() == 0
    assert [r["sequence"] for r in records] == list(range(len(records)))
    for index, query in enumerate(("q0", "q1", "q2")):
        mine = [r for r in records if r["query_index"] == index]
        assert {r["query"] for r in mine} == {query}
        assert [r["sample_index"] for r in mine if r["phase"] == "timed"] == list(range(7))
        assert [r["phase"] for r in mine][0] == "cold"


# ---- abort rules --------------------------------------------------------------------------------


@pytest.mark.parametrize("fail_call", [1, 2, 3, 6])  # cold, warm-up and timed positions
def test_a_non_200_response_aborts_instead_of_being_recorded(fail_call):
    with pytest.raises(bench.BenchmarkError, match="unexpected status 500"):
        collect(FakeClient(fail_on=fail_call))


def test_other_error_statuses_also_abort():
    with pytest.raises(bench.BenchmarkError, match="503"):
        collect(FakeClient(fail_on=4, status=503))


def test_a_changing_result_count_aborts():
    class Flaky(FakeClient):
        def get(self, url, params):
            response = super().get(url, params)
            if self.calls > 6:
                response._body["result_count"] = self.calls
            return response

    with pytest.raises(bench.BenchmarkError, match="result_count changed"):
        collect(Flaky())


# ---- percentiles and recomputation -----------------------------------------------------------


def test_nearest_rank_percentiles():
    values = [float(i) for i in range(1, 201)]
    assert [bench.percentile(values, p) for p in (50, 95, 99)] == [100.0, 190.0, 198.0]
    assert bench.percentile([5.0], 99) == 5.0
    assert bench.summarize([3.0, 1.0, 2.0])["p50"] == 2.0


def make_artifacts(tmp_path, run_records):
    results = [
        {"run": run, "catalog": {"products": 240, "documents": 240}, "explain": {}}
        for run in run_records
    ]
    records = [r for rs in run_records.values() for r in rs]
    protocol = {"runs": len(run_records), "queries": ["q0", "q1", "q2"]}
    raw = tmp_path / f"{EXPERIMENT}.samples.jsonl"
    header = {"experiment_id": EXPERIMENT, "source_tree_sha256": SOURCE["source_tree_sha256"]}
    sha = bench.write_samples(raw, header, records)
    report = bench.assemble_report(
        experiment_id=EXPERIMENT,
        source=SOURCE,
        environment={"generated_at_utc": "2026-10-01T00:00:00+00:00"},
        protocol=protocol,
        run_results=results,
        records=records,
        raw_name=raw.name,
        raw_sha=sha,
    )
    summary = tmp_path / f"{EXPERIMENT}.json"
    summary.write_text(json.dumps(report), encoding="utf-8")
    return summary, raw, report


def test_summary_recomputes_exactly_from_raw_samples_with_run_boundaries(tmp_path):
    runs = {1: collect(FakeClient(), run=1), 2: collect(FakeClient(), run=2)}
    summary, raw, report = make_artifacts(tmp_path, runs)
    assert bench.verify_artifacts(summary) == []
    _, records = bench.read_samples(raw)
    assert len(records) == 48 and {r["run"] for r in records} == {1, 2}
    for run_summary in report["runs"]:
        mine = [r for r in records if r["run"] == run_summary["run"] and r["phase"] == "timed"]
        for metric in bench.METRICS:
            ordered = sorted(r[metric] for r in mine)
            n = len(ordered)  # independent nearest-rank implementation
            expected = {
                p: round(ordered[max(1, math.ceil(q * n / 100)) - 1], 3)
                for p, q in (("p50", 50), ("p95", 95), ("p99", 99))
            }
            got = run_summary["overall"][metric]
            assert {k: got[k] for k in ("p50", "p95", "p99")} == expected and got["n"] == n
    pooled = [r for r in records if r["phase"] == "timed"]
    assert report["pooled_across_runs"]["n_per_metric"] == len(pooled) == 42
    # run-level and pooled figures differ in n: boundaries are preserved, not merged
    assert report["runs"][0]["timed_samples"] == 21 != report["pooled_across_runs"]["n_per_metric"]


def test_artifact_shares_experiment_id_and_source_fingerprint(tmp_path):
    summary, raw, report = make_artifacts(tmp_path, {1: collect(FakeClient())})
    header, records = bench.read_samples(raw)
    assert header["experiment_id"] == report["experiment_id"] == EXPERIMENT
    assert header["source_tree_sha256"] == report["source"]["source_tree_sha256"]
    assert {r["experiment_id"] for r in records} == {EXPERIMENT}
    assert (
        report["raw_samples"]["timed_samples"] == 21 and report["raw_samples"]["cold_samples"] == 3
    )
    # tampering with either side is detected
    original = raw.read_text(encoding="utf-8")
    raw.write_text(original.replace(EXPERIMENT, "other-experiment", 2), encoding="utf-8")
    assert bench.verify_artifacts(summary)


def test_the_raw_file_is_deterministic_jsonl(tmp_path):
    records = collect(FakeClient())
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    header = {"experiment_id": EXPERIMENT, "source_tree_sha256": "c" * 64}
    assert bench.write_samples(first, header, records) == bench.write_samples(
        second, header, list(reversed(records))
    )
    assert first.read_bytes() == second.read_bytes() and b"\r" not in first.read_bytes()
    lines = first.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["record"] == "header" and len(lines) == 1 + len(records)


# ---- outliers are flagged, never removed ----------------------------------------------------


def test_an_injected_extreme_sample_stays_in_raw_data_and_affects_summaries(tmp_path):
    runs = {1: collect(FakeClient(), run=1)}
    baseline = bench.summarize_records(runs[1])
    victim = next(
        r
        for r in runs[1]
        if r["phase"] == "timed" and r["query"] == "q1" and r["sample_index"] == 3
    )
    victim["request_ms"] = 98931.725
    group = [r for r in runs[1] if r["phase"] == "timed" and r["query"] == "q1"]
    bench.flag_outliers(group)
    summary, raw, report = make_artifacts(tmp_path, runs)
    _, records = bench.read_samples(raw)
    kept = [r for r in records if r["request_ms"] == 98931.725]
    assert (
        len(kept) == 1 and kept[0]["outlier"] is True and "request_ms" in kept[0]["outlier_metrics"]
    )
    run = report["runs"][0]
    assert run["overall"]["request_ms"]["max"] == 98931.725
    assert run["overall"]["request_ms"]["mean"] > baseline["overall"]["request_ms"]["mean"] * 10
    assert run["per_query"]["q1"]["timed"]["request_ms"]["n"] == 7  # nothing excluded
    assert run["timed_samples"] == 21
    assert [o["request_ms"] for o in run["flagged_outliers"]] == [98931.725]
    assert bench.verify_artifacts(summary) == []  # the summary still recomputes from raw data


def test_a_real_stall_is_flagged_but_remains_in_the_percentile_inputs():
    client = FakeClient(stall_on=12, stall=0.08)  # call 12 is a timed sample of the first query
    records = collect(client, queries=("q0",), warmup=2, samples=10)
    stalled = max((r for r in records if r["phase"] == "timed"), key=lambda r: r["request_ms"])
    assert stalled["request_ms"] >= 80 and stalled["outlier"] is True
    summary = bench.summarize_records(records)
    assert summary["overall"]["request_ms"]["n"] == 10
    assert summary["overall"]["request_ms"]["max"] == round(stalled["request_ms"], 3)
    assert any(o["sequence"] == stalled["sequence"] for o in summary["flagged_outliers"])


def test_the_outlier_rule_is_relative_to_the_group_median():
    group = [{"request_ms": 5.0, "lexical_ms": 1.0} for _ in range(9)]
    group.append({"request_ms": 51.0, "lexical_ms": 1.0})  # 10.2 x the median
    bench.flag_outliers(group)
    assert [g["outlier"] for g in group] == [False] * 9 + [True]
    assert group[-1]["outlier_metrics"] == ["request_ms"]
    assert bench.OUTLIER_FACTOR == 10
