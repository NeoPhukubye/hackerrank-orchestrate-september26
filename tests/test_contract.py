"""Basic contract tests for the Buy or Wait solution."""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "code"))


def _req_amount(rid):
    with open(os.path.join(REPO, "dataset/requests.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["request_id"] == rid:
                return float(row["requested_amount"])
    raise KeyError(rid)


def test_output_exists_and_valid():
    out_path = os.path.join(REPO, "output.csv")
    assert os.path.exists(out_path), "output.csv must exist"
    with open(out_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 250
    expected = ["request_id", "amount_safe_to_pay", "affordability_status",
                "recommended_payment_method", "payment_plan",
                "earliest_date_for_full_payment", "spending_changes_needed",
                "decision_explanation"]
    assert list(rows[0].keys()) == expected
    for r in rows:
        amt = float(r["amount_safe_to_pay"])
        req_amt = _req_amount(r["request_id"])
        assert 0 <= amt <= req_amt
        assert r["affordability_status"] in {"affordable_now", "affordable_with_plan",
                                              "affordable_later", "not_affordable"}
        assert r["recommended_payment_method"] in {"full_payment", "partial_payment",
                                                   "installments", "wait", "not_recommended"}
