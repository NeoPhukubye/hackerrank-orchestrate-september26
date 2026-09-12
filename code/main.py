"""Buy or Wait? -- AI-powered financial decision agent.

Entry point for the HackerRank Orchestrate challenge.  Reads the participant
dataset, decides for every request whether the user should pay in full, pay
partially, use installments, wait, or not proceed, and writes ``output.csv``.

Pipeline
--------
1. **Multimodal ingestion** -- ``dataset/images.csv`` points at PNG receipts
   and statements; the amounts are recovered by the vision pass in
   ``code/vision/extract_images.py`` and cached in
   ``dataset/image_amounts.json``.
2. **Message understanding** -- ``code/messages/extract_messages.py`` turns
   ``dataset/messages.csv`` into structured amendments (salary changes, ended
   income, cancellations, refunds) in ``dataset/message_actions.json``.
3. **Deterministic reconstruction** -- ``code/engine.py`` normalises events,
   converts currencies with the dated rates, detects recurrence, and runs the
   90-day balance forecast.
4. **Plan generation and ranking** -- safe plans only, ranked by the ordering
   in ``problem_statement.md``.
5. **Grounded explanation synthesis** -- the explanation is rendered from the
   same numbers the decision used.

Usage
-----
    python3 code/main.py                     # writes ./output.csv
    python3 code/main.py --data dataset --out output.csv
    python3 code/main.py --samples           # score against the solved samples
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (  # noqa: E402
    HORIZON_DAYS,
    Datasets,
    Plan,
    PlannedChange,
    UserState,
    build_plans,
    build_user_state,
    earliest_safe_full_payment,
    fmt_amount,
    fmt_money,
    full_amount_of,
    is_cash_event,
    max_safe_charge,
    min_balance_reached,
    parse_date,
    rank_plans,
    resolve_amount,
    to_float,
)

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

METHOD_LABEL = {
    "full_payment": "full_payment",
    "partial_payment": "partial_payment",
    "installments": "installments",
    "wait": "wait",
    "not_recommended": "not_recommended",
}


# --------------------------------------------------------------------------
# Future expense rows (used for spending-change candidates)
# --------------------------------------------------------------------------

def future_expense_rows(ds: Datasets, user_id: str, request_date: date) -> Dict[str, List[dict]]:
    """Pending / scheduled outflows still ahead of the request date, grouped
    by category so a spending change can name a real ``event_id``."""
    out: Dict[str, List[dict]] = {}
    for row in ds.events.get(user_id, []):
        if row.get("direction") != "debit":
            continue
        if not is_cash_event(row):
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when is None or when < request_date:
            continue
        if resolve_amount(ds, row) is None:
            continue
        out.setdefault(row["category"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: parse_date(r.get("settlement_date")) or date.max)
    return out


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

def decide(ds: Datasets, request: dict, state: UserState) -> dict:
    request_date = parse_date(request["request_date"])
    requested = full_amount_of(request)
    deadline = parse_date(request.get("desired_completion_date") or "")

    rows = future_expense_rows(ds, request["user_id"], request_date)
    options = ds.options.get(request["request_id"], [])

    # ``amount_safe_to_pay`` is the free cash the user can commit today while
    # keeping the minimum protected across the whole 90-day forecast, before
    # any optional spending changes, capped at the requested amount.
    # The largest charge whose *own* effect on the 90-day path still leaves the
    # minimum intact. Paying more than the horizon's free cash would push a
    # later commitment below the minimum.
    safe_today = max(0.0, min(
        max_safe_charge(state, request_date, request_date, requested),
        min_balance_reached(state, request_date, HORIZON_DAYS) - state.minimum,
        requested,
    ))
    earliest_full = earliest_safe_full_payment(state, requested, request_date)

    plans = build_plans(ds, state, request, options, rows, request_date)
    ranked = rank_plans(plans, request_date, deadline)

    if ranked:
        return _render_plan(
            ranked[0], state, request, request_date, requested, safe_today, earliest_full, rows
        )

    return _render_unaffordable(state, request, request_date, requested, safe_today, earliest_full)


def _render_plan(
    plan: Plan,
    state: UserState,
    request: dict,
    request_date: date,
    requested: float,
    safe_today: float,
    earliest_full: Optional[date],
    rows: Dict[str, List[dict]],
) -> dict:
    plan_str = "|".join(
        f"{when.isoformat()}:{fmt_amount(amount)}" for when, amount in plan.payments
    ) or "none"

    # ``affordable_now`` means the full amount is safe today, so the earliest
    # safe date for a single full payment is by definition the request date.
    if plan.status == "affordable_now":
        earliest_full = request_date

    # Populated whenever a full payment is forecast safe inside the horizon;
    # left empty when it never becomes safe within the forecast period.
    earliest_str = earliest_full.isoformat() if earliest_full else ""

    status = plan.status
    if plan.kind == "full_payment":
        explain = _explain_full(state, requested, plan, request_date, rows)
    elif plan.kind == "installments":
        explain = _explain_installments(state, plan, rows)
    elif plan.kind == "partial_payment":
        explain = _explain_partial(state, plan)
    else:  # wait
        explain = _explain_wait(state, plan, request_date)

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": fmt_amount(min(safe_today, requested)),
        "affordability_status": status,
        "recommended_payment_method": METHOD_LABEL[plan.kind],
        "payment_plan": plan_str,
        "earliest_date_for_full_payment": earliest_str,
        "spending_changes_needed": _render_changes(plan) or "none",
        "decision_explanation": explain,
    }


def _render_unaffordable(
    state: UserState,
    request: dict,
    request_date: date,
    requested: float,
    safe_today: float,
    earliest_full: Optional[date],
) -> dict:
    currency = state.currency
    deadline = parse_date(request.get("desired_completion_date") or "")
    deadline_str = _pretty_date(deadline) if deadline else "the deadline"

    if earliest_full is not None:
        explanation = (
            f"Do not proceed with the {currency} {fmt_money(requested)} request. "
            f"Although {currency} {fmt_money(safe_today)} is available today, the full "
            f"amount cannot be completed safely within the {HORIZON_DAYS}-day forecast."
        )
    else:
        explanation = (
            f"Do not make this payment by {deadline_str}. None of the available options "
            f"keeps the {currency} {fmt_money(state.minimum)} minimum protected."
        )

    return {
        "request_id": request["request_id"],
        "amount_safe_to_pay": fmt_amount(min(safe_today, requested)),
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": earliest_full.isoformat() if earliest_full else "",
        "spending_changes_needed": "none",
        "decision_explanation": explanation,
    }


# --------------------------------------------------------------------------
# Explanation synthesis (grounded in the numbers actually used)
# --------------------------------------------------------------------------

def _pretty_date(d: Optional[date]) -> str:
    if d is None:
        return ""
    return f"{d.day} {d.strftime('%B %Y')}"


def _leftover(state: UserState, plan: Plan) -> float:
    start = plan.payments[0][0] if plan.payments else date.today()
    charges = [(when, -amount) for when, amount in plan.payments]
    return max(min_balance_reached(state, start, HORIZON_DAYS, charges, plan.changes), state.minimum)


def _describe_changes(plan: Plan, rows: Dict[str, List[dict]]) -> str:
    """Human phrase for the spending changes, e.g. "Stop the family streaming
    plan" or "Reduce the weekend food delivery to IDR 665,950"."""
    parts: List[str] = []
    for change in plan.changes:
        row = _find_row(change, rows)
        if row is None:
            continue
        desc = (row.get("description") or row.get("category", "")).strip()
        if change.new_amount <= 0:
            parts.append(f"Stop the {desc.lower()}")
        else:
            parts.append(f"Reduce the {desc.lower()}")
    return " and ".join(parts)


def _find_row(change: PlannedChange, rows: Dict[str, List[dict]]) -> Optional[dict]:
    for category_rows in rows.values():
        for row in category_rows:
            if row["event_id"] == change.event_id:
                return row
    return None


def _render_changes(plan: Plan) -> str:
    pieces = []
    for change in plan.changes:
        if change.new_amount <= 0:
            pieces.append(f"stop:{change.event_id}")
        else:
            pieces.append(f"reduce_to:{change.event_id}:{fmt_amount(change.new_amount)}")
    return "|".join(pieces)


def _explain_full(
    state: UserState,
    requested: float,
    plan: Plan,
    request_date: date,
    rows: Dict[str, List[dict]],
) -> str:
    currency = state.currency
    leftover = _leftover(state, plan)
    when = plan.payments[0][0]

    if plan.changes:
        prefix = _describe_changes(plan, rows)
        return (
            f"{prefix}, then pay {currency} {fmt_money(requested)} today. This leaves at "
            f"least {currency} {fmt_money(leftover)} available."
        )
    if when == request_date:
        return (
            f"Pay {currency} {fmt_money(requested)} today. This leaves at least "
            f"{currency} {fmt_money(leftover)} available over the next {HORIZON_DAYS} days."
        )
    return (
        f"Pay {currency} {fmt_money(requested)} in full on {_pretty_date(when)}. Paying "
        f"earlier would take the balance below the {currency} {fmt_money(state.minimum)} minimum."
    )


def _explain_installments(state: UserState, plan: Plan, rows: Dict[str, List[dict]]) -> str:
    currency = state.currency
    count = len(plan.payments)
    amount = plan.payments[0][1]
    first = _pretty_date(plan.payments[0][0])
    leftover = _leftover(state, plan)
    lead = "Use"
    if plan.changes:
        lead = f"{_describe_changes(plan, rows)}, then use"
    return (
        f"{lead} {count} installments of {currency} {fmt_money(amount)}, starting {first}. "
        f"This leaves at least {currency} {fmt_money(leftover)} available."
    )


def _explain_partial(state: UserState, plan: Plan) -> str:
    currency = state.currency
    head = plan.payments[0][1]
    tail = plan.payments[1][1]
    tail_date = _pretty_date(plan.payments[1][0])
    return (
        f"Pay {currency} {fmt_money(head)} today and the remaining {currency} "
        f"{fmt_money(tail)} on {tail_date}. This completes the full request and keeps the "
        f"{currency} {fmt_money(state.minimum)} minimum protected."
    )


def _explain_wait(state: UserState, plan: Plan, request_date: date) -> str:
    currency = state.currency
    requested = plan.payments[0][1]
    when = plan.payments[0][0]
    if when == request_date:
        return (
            f"Pay {currency} {fmt_money(requested)} in full today. This leaves at least "
            f"{currency} {fmt_money(state.minimum)} available."
        )
    return (
        f"Wait until {_pretty_date(when)}, then pay {currency} {fmt_money(requested)} in full. "
        f"Paying sooner would put the {currency} {fmt_money(state.minimum)} minimum at risk."
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def solve(ds: Datasets, requests: Sequence[dict]) -> List[dict]:
    results: List[dict] = []
    cache: Dict[str, UserState] = {}
    for request in requests:
        request_date = parse_date(request["request_date"])
        key = f"{request['user_id']}::{request_date.isoformat()}"
        if key not in cache:
            cache[key] = build_user_state(ds, request["user_id"], request_date)
        results.append(decide(ds, request, cache[key]))
    return results


def write_output(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def validate(rows: Sequence[dict], requests: Sequence[dict]) -> List[str]:
    """Deterministic post-checks demanded by the problem statement."""
    problems: List[str] = []
    wanted = {r["request_id"] for r in requests}
    seen = {r["request_id"] for r in rows}
    missing = wanted - seen
    if missing:
        problems.append(f"missing {len(missing)} request rows, e.g. {sorted(missing)[:3]}")

    allowed_status = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
    allowed_method = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
    by_id = {r["request_id"]: r for r in requests}

    for row in rows:
        rid = row["request_id"]
        req = by_id.get(rid)
        if req is None:
            continue
        requested = to_float(req["requested_amount"])
        try:
            safe = float(row["amount_safe_to_pay"])
        except ValueError:
            problems.append(f"{rid}: non-numeric amount_safe_to_pay")
            continue
        if not (0 <= safe <= requested + 1e-6):
            problems.append(f"{rid}: amount_safe_to_pay {safe} outside [0, {requested}]")
        if row["affordability_status"] not in allowed_status:
            problems.append(f"{rid}: bad status {row['affordability_status']}")
        if row["recommended_payment_method"] not in allowed_method:
            problems.append(f"{rid}: bad method {row['recommended_payment_method']}")
        if row["affordability_status"] == "affordable_now" and \
                row["earliest_date_for_full_payment"] != req["request_date"]:
            problems.append(f"{rid}: affordable_now must have earliest == request_date")
    return problems


def score_against_samples(ds: Datasets, path: str, dump_path: str = "") -> None:
    """Score the engine on the 25 solved examples shipped with the dataset."""
    with open(path, encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))

    keep = {
        "request_id", "user_id", "request_date", "request_type",
        "requested_amount", "desired_completion_date",
        "allows_partial_payment", "request_text",
    }
    requests = [{k: v for k, v in row.items() if k in keep} for row in samples]

    rows = solve(ds, requests)
    if dump_path:
        write_output(dump_path, rows)
    preds = {r["request_id"]: r for r in rows}
    status_ok = method_ok = plan_ok = 0
    amount_mae = 0.0
    misses: List[str] = []

    for row in samples:
        rid = row["request_id"]
        pred = preds[rid]
        status_ok += pred["affordability_status"] == row["affordability_status"]
        method_ok += pred["recommended_payment_method"] == row["recommended_payment_method"]
        plan_ok += pred["payment_plan"] == row["payment_plan"]
        try:
            amount_mae += abs(float(pred["amount_safe_to_pay"]) - float(row["amount_safe_to_pay"]))
        except ValueError:
            pass
        if pred["affordability_status"] != row["affordability_status"] or \
                pred["recommended_payment_method"] != row["recommended_payment_method"]:
            misses.append(
                f"  {rid}: pred {pred['affordability_status']}/{pred['recommended_payment_method']}"
                f" vs truth {row['affordability_status']}/{row['recommended_payment_method']}"
            )

    n = len(samples)
    print(f"samples scored      : {n}")
    print(f"status accuracy     : {status_ok / n:.1%}")
    print(f"method accuracy     : {method_ok / n:.1%}")
    print(f"exact plan match    : {plan_ok / n:.1%}")
    print(f"amount_safe MAE     : {amount_mae / n:,.2f}")
    if misses:
        print("mismatches:")
        print("\n".join(misses))
    if dump_path:
        print(f"sample predictions written: {dump_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Buy or Wait? decision agent")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data", default=os.path.join(here, "dataset"),
                        help="dataset directory (default: <repo>/dataset)")
    parser.add_argument("--out", default=os.path.join(here, "output.csv"),
                        help="output CSV path (default: <repo>/output.csv)")
    parser.add_argument("--samples", action="store_true",
                        help="score against dataset/sample_requests.csv")
    parser.add_argument("--dump", default=None,
                        help="write sample predictions here (use with --samples), "
                             "so code/evaluation/main.py can score them")
    parser.add_argument("--mirror-template", action="store_true",
                        help="also refresh dataset/output.csv")
    args = parser.parse_args(argv)

    ds = Datasets.load(args.data)

    if args.samples:
        dump = args.dump or "/tmp/sample_predictions.csv"
        score_against_samples(ds, os.path.join(args.data, "sample_requests.csv"), dump)
        return 0

    with open(os.path.join(args.data, "requests.csv"), encoding="utf-8", newline="") as handle:
        requests = list(csv.DictReader(handle))

    rows = solve(ds, requests)
    write_output(args.out, rows)

    if args.mirror_template:
        write_output(os.path.join(args.data, "output.csv"), rows)

    problems = validate(rows, requests)
    if problems:
        print(f"validation: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems[:15]:
            print(f"  - {problem}", file=sys.stderr)
    else:
        print(f"validation: {len(rows)}/{len(requests)} rows OK")

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["recommended_payment_method"]] = counts.get(row["recommended_payment_method"], 0) + 1
    print("methods:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
