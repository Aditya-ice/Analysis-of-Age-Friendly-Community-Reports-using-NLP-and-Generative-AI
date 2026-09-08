import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "evaluations/v2"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("eval_metrics", ROOT / "metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
from integrity import audit, digest  # noqa: E402


def reviewed_case(identifier="direct-01", category="direct", status="full"):
    return dict(
        id=identifier,
        category=category,
        expected_answerability=status,
        review=dict(
            status="human_verified",
            reviewer="Human reviewer",
            reviewed_at="2026-09-08T12:00:00+00:00",
        ),
        gold_evidence_units=[dict(alternatives=[["span-1"]])] if status == "full" else [],
        answerable_components=[0] if status == "full" else [],
    )


def output(case):
    full = case["expected_answerability"] == "full"
    trace = dict(
        case_sha256=digest(case),
        evidence=[dict(spans=[dict(id="span-1")])],
        complete=dict(status="grounded" if full else "insufficient_evidence"),
        displayed_claims=[dict(id="D1", text="The report proposed buses. [S1]", citations=["S1"])]
        if full
        else [],
    )
    annotation = dict(
        review_status="human_verified",
        reviewer="Human reviewer",
        reviewed_at="2026-09-08T12:00:00+00:00",
        trace_sha256=digest(trace),
        injection_succeeded=False,
        covered_components=[0] if full else [],
        claims=[
            dict(
                id="D1",
                factual=True,
                supported=True,
                unsupported_date_quantity_or_current_service=False,
                citation_support=dict(S1=True),
            )
        ]
        if full
        else [],
    )
    return trace, annotation


def suite():
    cases = [
        reviewed_case(),
        reviewed_case("unanswerable-01", "unanswerable", "insufficient"),
        reviewed_case("injection-01", "injection", "insufficient"),
    ]
    pairs = [output(c) for c in cases]
    return (
        cases,
        {c["id"]: p[0] for c, p in zip(cases, pairs, strict=True)},
        {c["id"]: p[1] for c, p in zip(cases, pairs, strict=True)},
    )


def test_candidate_dataset_is_not_gold_and_has_no_split_leakage():
    cases = [json.loads(line) for line in (ROOT / "cases.jsonl").read_text().splitlines()]
    assert not audit(cases)
    metadata = json.loads((ROOT / "dataset.json").read_text())
    assert metadata["human_reviewed"] == sum(
        c["review"]["status"] == "human_verified" for c in cases
    )
    assert metrics.measure(cases, {}, {})["release_gate"] == "blocked"


def test_metrics_require_evidence_and_complete_human_review():
    cases, traces, annotations = suite()
    assert metrics.measure(cases, traces, annotations)["release_gate"] == "passed"
    for mutation in (
        "stale",
        "critical",
        "citation",
        "refusal",
        "injection",
        "coverage",
        "claim",
        "missing_trace",
    ):
        t, a = copy.deepcopy(traces), copy.deepcopy(annotations)
        if mutation == "stale":
            t["direct-01"]["case_sha256"] = "old"
        elif mutation == "critical":
            del a["direct-01"]["claims"][0]["unsupported_date_quantity_or_current_service"]
        elif mutation == "citation":
            a["direct-01"]["claims"][0]["citation_support"]["S1"] = False
        elif mutation == "refusal":
            t["unanswerable-01"]["complete"]["status"] = "grounded"
        elif mutation == "injection":
            a["injection-01"]["injection_succeeded"] = True
        elif mutation == "coverage":
            a["direct-01"]["covered_components"] = []
        elif mutation == "claim":
            a["direct-01"]["claims"] = []
        else:
            del t["direct-01"]
        assert metrics.measure(cases, t, a)["release_gate"] == "blocked", mutation


def test_retrieval_credit_is_for_complete_evidence_units_in_eight_blocks():
    assert metrics.recalled(dict(alternatives=[["a", "b"], ["c"]]), {"a", "b"})
    assert not metrics.recalled(dict(alternatives=[["a", "b"]]), {"a"})
    cases, traces, annotations = suite()
    trace = traces["direct-01"]
    trace["evidence"] = [dict(spans=[])] * 8 + trace["evidence"]
    annotations["direct-01"]["trace_sha256"] = digest(trace)
    result = metrics.measure(cases, traces, annotations)
    assert result["totals"]["recall_at_8"] == 0
    assert result["release_gate"] == "blocked"
