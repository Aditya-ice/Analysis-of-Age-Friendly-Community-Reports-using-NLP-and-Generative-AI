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
