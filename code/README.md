# Buy or Wait? — Solution Code

An AI-powered financial decision agent for the **HackerRank Orchestrate
September 2026** challenge. For every request it decides whether the user
should pay in full, pay partially, use installments, wait, or not proceed, and
writes the required `output.csv`.

## Quick Start

```bash
# 1. Deterministic run — produces ./output.csv (no API keys needed)
python3 code/main.py

# 2. Verify the output against every structural rule
python3 code/evaluation/main.py

# 3. Score the engine against the 25 solved examples
python3 code/main.py --samples
```

`output.csv` is written to the repository root. `python3 code/main.py
--mirror-template` also refreshes `dataset/output.csv`.

Re-running `code/main.py` after the two extraction caches exist costs **zero
tokens** and produces byte-identical results.

## Design: Hybrid Deterministic–Agentic Pipeline

```
  dataset/media/images/*.png ─┐
  dataset/images.csv          │
                              ├─► code/vision/extract_images.py   (model)  ─► dataset/image_amounts.json
  dataset/messages.csv ───────┴─► code/messages/extract_messages.py (model)  ─► dataset/message_actions.json
                                              │
                                              ▼
                              code/engine.py  — deterministic reconstruction
                              · currency conversion via dated exchange rates
                              · recurrence detection over settled history
                              · 90-day daily balance forecast
                              · constraint verification (min balance, bounds)
                                              │
                                              ▼
                              code/main.py    — plan ranking + CSV writer
                                              │
                                              ▼
                                      output.csv
```

The split is deliberate:

- **Models read the unstructured evidence.** Receipt totals and payroll
  amendments are figures buried in prose and pixels, so a vision/language pass
  extracts them once and caches the result.
- **Code makes every monetary decision.** Currency conversion, recurrence
  detection, the daily balance forecast, and plan ranking are plain arithmetic
  in `code/engine.py`. A language model never performs the cash-flow maths, so
  the recommendation is reproducible and auditable.

## Files

| Path | Role |
|---|---|
| `code/main.py` | Entry point: orchestrates the pipeline and writes `output.csv` |
| `code/engine.py` | Deterministic core: FX, recurrence, forecast, plan selection |
| `code/evaluation/main.py` | Contract checks + scoring against the solved examples |
| `code/evaluation/usage_report.md` | Token usage and cost for the final full-dataset run |
| `code/vision/extract_images.py` | Reads the receipt/statement images for the 16 blank amounts |
| `code/messages/extract_messages.py` | Turns message prose into structured financial amendments |

## How a Decision Is Made

1. **Reconstruct the financial state.** `settled` rows form the history;
   `pending` and `scheduled` rows are the foreground ledger; `cancelled`,
   `failed`, and `unrealized` rows never move cash. Pending credits, bonuses,
   commissions, refunds, and investment gains are excluded until they settle.
   Unrealized investment value is never treated as spendable cash.
2. **Recover missing amounts from images.** When a financial event has a blank
   `amount`, the figure is read from the image linked through `images.csv`
   (`related_event_id`). A blank amount is **never** treated as zero;
   `python3 code/vision/extract_images.py --verify` proves all 16 are resolved.
3. **Convert currency on the settlement date** using `exchange_rates.csv`,
   taking the most recent rate at or before that date for the stated pair.
4. **Detect recurrence** only where the settled history supports it (at least
   two observations inside a 45-day window), snapping each series to its true
   cadence. Series that the ledger explicitly ends — for example a row
   described as the *final* payroll — are not extrapolated.
5. **Forecast 90 days** day by day:
   `Balance(t) = Balance(t-1) + Inflow(t) - Essential(t) - Committed(t) - Flexible(t)`,
   enforcing the safety invariant at every step:

   ```
   min over t in [request_date, request_date + 90] of Balance(t) >= minimum_balance_to_keep
   ```

6. **Generate only safe plans**, each restricted to methods in the user's
   `payment_methods_user_will_consider`. Installment plans must exactly match a
   supplied option and respect `max_installment_months`. Partial payment is
   offered only when the request allows it and the remainder is due on or
   before `desired_completion_date`.
7. **Rank the safe plans** in the order the statement prescribes:
   completes by the deadline → no spending changes → lowest total cost →
   earliest start → fewest payments → lowest `payment_option_id`.

### Spending Changes

Only flexible, recurring, non-protected expenses may be changed, and a change
always names a concrete `event_id`:

```text
stop:<event_id>
reduce_to:<event_id>:<new_amount>
```

Stopping and reducing the same event are mutually exclusive, so at most one
change is made per category.

## Verification

`python3 code/evaluation/main.py` checks every rule that can be verified
without the hidden answers:

- exactly one row per `request_id` in `dataset/requests.csv`, with the required
  columns in the required order
- `0 <= amount_safe_to_pay <= requested_amount`
- valid `affordability_status` and `recommended_payment_method` values
- `affordable_now` implies `earliest_date_for_full_payment == request_date`
- partial plans carry exactly two payments that sum to `requested_amount`,
  start on `request_date`, and start with exactly `amount_safe_to_pay`
- installment plans match a supplied payment option exactly
- spending changes reference flexible, non-protected events, at most three,
  never the same event twice

Current status on the final 250-request run: **all checks pass**.

Measured against the 25 solved examples in `dataset/sample_requests.csv`:

| Metric | Score |
|---|---|
| `affordability_status` accuracy | 84.0% |
| `recommended_payment_method` accuracy | 88.0% |
| Exact `payment_plan` match | 64.0% |

Reproduce with:

```bash
python3 code/main.py --samples
python3 code/evaluation/main.py --predictions /tmp/sample_predictions.csv
```

## Requirements

- Python 3.9+ (standard library only — no third-party packages to install)
- Run from the repository root so the default `dataset/` path resolves

`code/main.py`, `code/engine.py`, and `code/evaluation/main.py` need no network
access and no API key. The two extraction scripts document the model passes
that produced the cached evidence; their outputs are already committed at
`dataset/image_amounts.json` and `dataset/message_actions.json`, so the pipeline
is fully reproducible offline.
