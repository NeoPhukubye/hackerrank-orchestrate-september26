"""Scoring harness for the Buy or Wait? agent.

Scores a prediction file against the solved examples and runs the
deterministic contract checks that ``problem_statement.md`` requires, so a run
can be self-verified before submission.

Usage
-----
    python3 code/evaluation/main.py
    python3 code/evaluation/main.py --predictions output.csv --data dataset
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALLOWED_STATUS = {
    "affordable_now", "affordable_with_plan", "affordable_later", "not_affordable",
}
ALLOWED_METHOD = {
    "full_payment", "partial_payment", "installments", "wait", "not_recommended",
}
FLEXIBLE = {"stoppable", "reducible", "reducible_or_stoppable"}

OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]


def load(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def score(predictions: Sequence[dict], truth: Sequence[dict]) -> Dict[str, float]:
    """Accuracy on the decision fields, plus MAE and relative error on amount."""
    preds = {r["request_id"]: r for r in predictions}
    status_ok = method_ok = plan_ok = 0
    abs_err = rel_err = 0.0
    n = 0
    for target in truth:
        pred = preds.get(target["request_id"])
        if pred is None:
            continue
        n += 1
        status_ok += pred["affordability_status"] == target["affordability_status"]
        method_ok += pred["recommended_payment_method"] == target["recommended_payment_method"]
        plan_ok += pred["payment_plan"] == target["payment_plan"]
        try:
            p = float(pred["amount_safe_to_pay"])
            t = float(target["amount_safe_to_pay"])
        except ValueError:
            continue
        abs_err += abs(p - t)
        if t:
            rel_err += abs(p - t) / abs(t)
    if not n:
        return {}
    return {
        "samples": n,
        "status_accuracy": status_ok / n,
        "method_accuracy": method_ok / n,
        "plan_accuracy": plan_ok / n,
        "amount_mae": abs_err / n,
        "amount_mean_relative_error": rel_err / n,
    }


def contract_checks(predictions: Sequence[dict], data_dir: str) -> List[str]:
    """Every structural rule the grader can verify without the hidden answers."""
    requests = {r["request_id"]: r for r in load(os.path.join(data_dir, "requests.csv"))}
    events = {e["event_id"]: e for e in load(os.path.join(data_dir, "financial_events.csv"))}
    profiles = {p["user_id"]: p for p in load(os.path.join(data_dir, "financial_profiles.csv"))}

    options: Dict[str, List[dict]] = {}
    for option in load(os.path.join(data_dir, "request_payment_options.csv")):
        options.setdefault(option["request_id"], []).append(option)

    problems: List[str] = []

    missing = set(requests) - {r["request_id"] for r in predictions}
    if missing:
        problems.append(f"missing {len(missing)} request rows (e.g. {sorted(missing)[:3]})")

    if predictions and list(predictions[0].keys()) != OUTPUT_COLUMNS:
        problems.append(f"column mismatch: {list(predictions[0].keys())}")

    for row in predictions:
        rid = row["request_id"]
        request = requests.get(rid)
        if request is None:
            problems.append(f"{rid}: not a known request")
            continue

        requested = float(request["requested_amount"])
        try:
            safe = float(row["amount_safe_to_pay"])
        except ValueError:
            problems.append(f"{rid}: amount_safe_to_pay is not numeric")
            continue

        if not 0 <= safe <= requested + 1e-6:
            problems.append(f"{rid}: amount_safe_to_pay {safe} outside [0, {requested}]")
        if row["affordability_status"] not in ALLOWED_STATUS:
            problems.append(f"{rid}: invalid affordability_status")
        if row["recommended_payment_method"] not in ALLOWED_METHOD:
            problems.append(f"{rid}: invalid recommended_payment_method")

        if row["affordability_status"] == "affordable_now":
            if row["earliest_date_for_full_payment"] != request["request_date"]:
                problems.append(f"{rid}: affordable_now requires earliest == request_date")

        method = row["recommended_payment_method"]
        plan = row["payment_plan"]

        if method == "partial_payment":
            if row["affordability_status"] != "affordable_with_plan":
                problems.append(f"{rid}: partial_payment must be affordable_with_plan")
            if request["allows_partial_payment"].strip().lower() != "true":
                problems.append(f"{rid}: request does not allow partial payment")
            parts = plan.split("|")
            if len(parts) != 2:
                problems.append(f"{rid}: partial_payment needs exactly two payments")
            else:
                dates = [p.split(":")[0] for p in parts]
                amounts = [float(p.split(":")[1]) for p in parts]
                if abs(sum(amounts) - requested) > 1:
                    problems.append(f"{rid}: partial payments do not sum to requested")
                if dates[0] != request["request_date"]:
                    problems.append(f"{rid}: partial first payment must be on request_date")
                if abs(amounts[0] - safe) > 1e-6:
                    problems.append(f"{rid}: partial first payment must equal amount_safe_to_pay")
                if dates[1] > request["desired_completion_date"]:
                    problems.append(f"{rid}: partial second payment is after the deadline")

        if method == "installments":
            if plan == "none":
                problems.append(f"{rid}: installments needs a payment plan")
            else:
                parts = [(p.split(":")[0], float(p.split(":")[1])) for p in plan.split("|")]
                matched = False
                for option in options.get(rid, []):
                    if option["payment_method"] != "installments":
                        continue
                    if int(float(option["number_of_payments"])) != len(parts):
                        continue
                    if all(abs(a - float(option["payment_amount"])) < 0.01 for _, a in parts):
                        matched = True
                        break
                if not matched:
                    problems.append(f"{rid}: installment plan does not match a supplied option")

        if method == "not_recommended" and plan != "none":
            problems.append(f"{rid}: not_recommended must use plan 'none'")

        changes = row["spending_changes_needed"]
        if changes != "none":
            parts = changes.split("|")
            if len(parts) > 3:
                problems.append(f"{rid}: more than three spending changes")
            protect = set(
                c.strip() for c in
                profiles[request["user_id"]]["expense_categories_to_protect"].split("|")
                if c.strip()
            )
            changed_events: List[str] = []
            for part in parts:
                if part.startswith("stop:"):
                    event_id = part.split(":")[1]
                elif part.startswith("reduce_to:"):
                    fields = part.split(":")
                    if len(fields) != 3:
                        problems.append(f"{rid}: malformed reduce_to change")
                        continue
                    event_id = fields[1]
                    try:
                        float(fields[2])
                    except ValueError:
                        problems.append(f"{rid}: reduce_to amount is not numeric")
                else:
                    problems.append(f"{rid}: malformed spending change '{part}'")
                    continue
                event = events.get(event_id)
                if event is None:
                    problems.append(f"{rid}: unknown event {event_id}")
                    continue
                if event["flexibility"] not in FLEXIBLE:
                    problems.append(f"{rid}: {event_id} is not a flexible expense")
                if event["category"] in protect:
                    problems.append(f"{rid}: {event_id} is in a protected category")
                changed_events.append(event_id)
            if len(changed_events) != len(set(changed_events)):
                problems.append(f"{rid}: the same event is changed more than once")

    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score and verify Buy or Wait? output")
    parser.add_argument("--predictions", default=os.path.join(REPO, "output.csv"))
    parser.add_argument("--data", default=os.path.join(REPO, "dataset"))
    args = parser.parse_args(argv)

    predictions = load(args.predictions)
    print(f"predictions: {args.predictions} ({len(predictions)} rows)")
    print()

    print("== contract checks ==")
    problems = contract_checks(predictions, args.data)
    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        for problem in problems[:25]:
            print(f"  - {problem}")
    else:
        print("PASSED: all structural requirements satisfied")
    print()

    samples_path = os.path.join(args.data, "sample_requests.csv")
    if os.path.exists(samples_path):
        scores = score(predictions, load(samples_path))
        print("== scored against the solved examples ==")
        if scores:
            print(f"samples scored          : {scores['samples']}")
            print(f"status accuracy         : {scores['status_accuracy']:.1%}")
            print(f"method accuracy         : {scores['method_accuracy']:.1%}")
            print(f"exact plan match        : {scores['plan_accuracy']:.1%}")
            print(f"amount_safe MAE         : {scores['amount_mae']:,.2f}")
            print(f"amount mean rel. error  : {scores['amount_mean_relative_error']:.2%}")
        else:
            print("no overlapping request_ids in the prediction file")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
