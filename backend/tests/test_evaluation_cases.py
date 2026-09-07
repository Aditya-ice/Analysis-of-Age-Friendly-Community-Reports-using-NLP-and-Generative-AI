import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_evaluation_set_has_expected_coverage():
    cases = [
        json.loads(line)
        for line in (ROOT / "evaluations/cases.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(cases) == 40
    assert len({case["id"] for case in cases}) == 40
    assert Counter(case["category"] for case in cases) == {
        "direct": 20,
        "comparison": 8,
        "followup": 6,
        "unanswerable": 6,
    }
