import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluations/v2"))
from agent_review import save_batch, validate  # noqa: E402
from review_batches import batches  # noqa: E402


def test_offline_batch_versioning_and_ai_separation(tmp_path, monkeypatch):
    import json

    import agent_review

    root = Path(__file__).resolve().parents[2] / "evaluations/v2"
    cases = [
        c
        for line in (root / "cases.jsonl").read_text().splitlines()
        if (c := json.loads(line))["split"] == "development"
    ]
    metadata = json.loads((root / "dataset.json").read_text())
    catalog = {}
    monkeypatch.setattr(agent_review, "load", lambda: (cases, metadata, catalog))
    assert len(cases) == 80 and all(c["split"] == "development" for c in cases)
    group = batches(cases)[0]
    annotations = {
        c["id"]: dict(
            answerability="insufficient",
            decision="unresolved",
            finding="Synthetic test fixture only",
            limitations=["Not reviewed"],
            sources=[],
        )
        for c in group
    }
    path = save_batch(1, annotations, tmp_path)

    record = json.loads(path.read_text())
    validate(record, group, metadata, catalog)
    for field in ("dataset_sha256", "index_fingerprint"):
        changed = {**record, field: "stale"}
        with pytest.raises(ValueError):
            validate(changed, group, metadata, catalog)
    changed = copy.deepcopy(record)
    changed["reviews"][0]["case_sha256"] = "stale"
    with pytest.raises(ValueError):
        validate(changed, group, metadata, catalog)
    changed = copy.deepcopy(record)
    changed["reviews"][0]["supporting_spans"] = [{"span_id": "invented"}]
    with pytest.raises(ValueError):
        validate(changed, group, metadata, catalog)
    with pytest.raises(ValueError):
        save_batch(1, annotations, tmp_path)
    changed = copy.deepcopy(record)
    changed["reviews"][0]["case_id"] = "heldout-case"
    with pytest.raises(ValueError):
        validate(changed, group, metadata, catalog)
    assert record["review_type"] == "ai_reviewed"
    from integrity import reviewed

    assert not reviewed(record)
    for field in ("human_verified", "gold_evidence_units", "reviewer", "review"):
        with pytest.raises(ValueError):
            validate({**record, field: True}, group, metadata, catalog)
    for field, value in (("status", "approved"), ("method", "unknown")):
        with pytest.raises(ValueError):
            validate({**record, field: value}, group, metadata, catalog)


def test_agent_spot_check_is_version_bound_and_not_human_certification(tmp_path, monkeypatch):
    import json

    import agent_report

    root = Path(__file__).resolve().parents[2] / "evaluations/v2"
    cases = {
        case["id"]: case
        for line in (root / "cases.jsonl").read_text().splitlines()
        if (case := json.loads(line))["split"] == "development"
    }
    metadata = json.loads((root / "dataset.json").read_text())
    source = root / "agent-feedback/agent-manual-spot-check.json"
    destination = tmp_path / source.name
    destination.write_text(source.read_text())
    monkeypatch.setattr(agent_report, "DEFAULT_OUTPUT", tmp_path)
    agent_reviews = {}
    for path in sorted((root / "agent-feedback").glob("development-*.json")):
        for review in json.loads(path.read_text())["reviews"]:
            agent_reviews[review["case_id"]] = review
    assert agent_report.load_agent_spot_check(cases, agent_reviews, metadata) == 12
    record = json.loads(destination.read_text())
    record["reviews"][0]["source_revision_ids"] = ["stale"]
    destination.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="source revision"):
        agent_report.load_agent_spot_check(cases, agent_reviews, metadata)
