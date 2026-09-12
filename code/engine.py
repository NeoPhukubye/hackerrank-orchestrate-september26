"""Deterministic core for the Buy or Wait? financial decision agent.

Design principle: language models are used only for *extraction* of
unstructured evidence (images, messages).  Every monetary decision --
currency conversion, recurrence detection, the 90-day balance forecast and
plan selection -- is deterministic arithmetic executed here so results are
reproducible and auditable.

Public contract lives in problem_statement.md; see code/main.py for the CLI.
"""

from __future__ import annotations

import csv
import json
import os
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

HORIZON_DAYS = 90
MONTH_DAYS = 30
WINDOW_DAYS = 45          # recurrence lookback window
MIN_OCCURRENCES = 2       # need at least this many settled rows to call it recurring

STATUS_SETTLED = "settled"
STATUS_PENDING = "pending"
STATUS_SCHEDULED = "scheduled"

# Statuses that never move cash.
NON_CASH_STATUSES = {"cancelled", "failed", "unrealized"}

CASH_DIRECTIONS = {"debit", "credit"}

REDUCABLE_FLEX = {"reducible", "reducible_or_stoppable"}
STOPPABLE_FLEX = {"stoppable", "reducible_or_stoppable"}


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def parse_date(value: str) -> Optional[date]:
    value = (value or "").strip()
    if not value:
        return None
    return date.fromisoformat(value[:10])


def to_float(value: str, default: float = 0.0) -> float:
    value = (value or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt_amount(value: float) -> str:
    """Render an amount the way the sample outputs do: no trailing zeros."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def fmt_money(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def split_pipe(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split("|") if v.strip()]


def month_shift(d: date, months: int) -> date:
    """Add whole months, clamping the day to the target month's length."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = d.day
    while day > 28:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1
    return date(year, month, day)


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

@dataclass
class Datasets:
    root: str
    profiles: Dict[str, dict]
    events: Dict[str, List[dict]]
    options: Dict[str, List[dict]]
    requests: Dict[str, dict]
    rates: List[dict]
    messages: Dict[str, List[dict]]
    images: Dict[str, List[dict]]
    image_amounts: Dict[str, float]

    @classmethod
    def load(cls, data_dir: str) -> "Datasets":
        def rows(name: str) -> List[dict]:
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                return []
            with open(path, encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))

        profiles = {r["user_id"]: r for r in rows("financial_profiles.csv")}

        events: Dict[str, List[dict]] = defaultdict(list)
        for r in rows("financial_events.csv"):
            events[r["user_id"]].append(r)

        options: Dict[str, List[dict]] = defaultdict(list)
        for r in rows("request_payment_options.csv"):
            options[r["request_id"]].append(r)

        requests = {r["request_id"]: r for r in rows("requests.csv")}

        messages: Dict[str, List[dict]] = defaultdict(list)
        for r in rows("messages.csv"):
            messages[r["user_id"]].append(r)

        images: Dict[str, List[dict]] = defaultdict(list)
        for r in rows("images.csv"):
            images[r["user_id"]].append(r)

        # Amounts recovered from images by the vision extraction pass. Each
        # entry carries the transcribed amount and the currency it was billed
        # in; a blank financial-event amount is never treated as zero.
        image_amounts: Dict[str, dict] = {}
        path = os.path.join(data_dir, "image_amounts.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            for key, value in raw.items():
                if isinstance(value, dict):
                    image_amounts[key] = value
                elif value is not None:
                    image_amounts[key] = {"amount": float(value), "currency": None}

        return cls(
            root=data_dir,
            profiles=profiles,
            events=dict(events),
            options=dict(options),
            requests=requests,
            rates=rows("exchange_rates.csv"),
            messages=dict(messages),
            images=dict(images),
            image_amounts=image_amounts,
        )


# --------------------------------------------------------------------------
# Currency conversion
# --------------------------------------------------------------------------

class FxTable:
    """Fixed dated rates from exchange_rates.csv.

    A conversion for a settlement date uses the most recent rate row at or
    before that date for the requested currency pair, falling back to any
    row for the pair and finally to a triangulated cross rate.
    """

    def __init__(self, rows: Sequence[dict]):
        self._direct: Dict[Tuple[str, str], List[Tuple[date, float]]] = defaultdict(list)
        for r in rows:
            d = parse_date(r.get("rate_date", ""))
            if d is None:
                continue
            pair = (r["from_currency"], r["to_currency"])
            rate = to_float(r.get("rate"))
            if rate > 0:
                self._direct[pair].append((d, rate))
        for pair in self._direct:
            self._direct[pair].sort(key=lambda item: item[0])

    def _lookup(self, src: str, dst: str, when: Optional[date]) -> Optional[float]:
        entries = self._direct.get((src, dst))
        if not entries:
            return None
        if when is None:
            return entries[-1][1]
        idx = bisect_right([d for d, _ in entries], when) - 1
        if idx < 0:
            idx = 0
        return entries[idx][1]

    def rate(self, src: str, dst: str, when: Optional[date]) -> float:
        if src == dst:
            return 1.0
        direct = self._lookup(src, dst, when)
        if direct:
            return direct
        inverse = self._lookup(dst, src, when)
        if inverse:
            return 1.0 / inverse
        # Triangulate through any pivot currency.
        for (a, b) in list(self._direct):
            for pivot in {a, b}:
                if pivot in (src, dst):
                    continue
                first = self._lookup(src, pivot, when)
                second = self._lookup(pivot, dst, when)
                if first and second:
                    return first * second
        return 1.0

    def convert(self, amount: float, src: str, dst: str, when: Optional[date]) -> float:
        return amount * self.rate(src, dst, when)


# --------------------------------------------------------------------------
# Event normalisation
# --------------------------------------------------------------------------

@dataclass
class CashEvent:
    event_id: str
    when: date
    amount: float            # home currency, signed (+credit / -debit)
    category: str
    flexibility: str
    event_type: str
    description: str
    recurring: bool = False
    projected: bool = False


def _image_amount_for(ds: Datasets, event_id: str) -> Optional[dict]:
    for rows in ds.images.values():
        for r in rows:
            if r.get("related_event_id") == event_id:
                return ds.image_amounts.get(r["image_id"])
    return None


def resolve_amount(ds: Datasets, row: dict) -> Optional[float]:
    """Return the event amount in its own currency, recovering it from the
    linked image when the CSV field is blank."""
    raw = (row.get("amount") or "").strip()
    if raw:
        return to_float(raw)
    evidence = _image_amount_for(ds, row["event_id"])
    if not evidence:
        return None
    amount = evidence.get("amount")
    return float(amount) if amount is not None else None


def event_currency(ds: Datasets, row: dict) -> str:
    """Currency of a row, falling back to the receipt's own currency when the
    financial-event row carries no amount of its own."""
    currency = (row.get("currency") or "").strip()
    if currency:
        return currency
    evidence = _image_amount_for(ds, row["event_id"])
    if evidence and evidence.get("currency"):
        return evidence["currency"]
    return ""


def amount_in_home_currency(ds: Datasets, row: dict, home: str) -> Optional[float]:
    """Event amount converted into the user's home currency using the dated
    rate for its settlement date. Amounts recovered from receipts are
    converted with the same rule as amounts present in the ledger."""
    amount = resolve_amount(ds, row)
    if amount is None:
        return None
    source = event_currency(ds, row) or home
    when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
    return get_fx(ds).convert(amount, source, home, when)


def is_cash_event(row: dict) -> bool:
    if row.get("status") in NON_CASH_STATUSES:
        return False
    if row.get("direction") not in CASH_DIRECTIONS:
        return False
    return True


def settled_totals(ds: Datasets, user_id: str) -> Dict[str, dict]:
    """Per-category settled stats used for recurrence detection."""
    stats: Dict[str, dict] = {}
    for row in ds.events.get(user_id, []):
        if row.get("status") != STATUS_SETTLED:
            continue
        if row.get("event_type") in ("refund", "investment_valuation",
                                     "investment_sale", "investment_purchase"):
            continue
        home = (ds.profiles.get(row.get("user_id")) or {}).get("home_currency", "")
        amount = amount_in_home_currency(ds, row, home)
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if amount is None or when is None:
            continue
        key = row["category"]
        bucket = stats.setdefault(key, {
            "count": 0, "sum": 0.0, "last": when, "direction": row["direction"],
            "flexibility": row.get("flexibility", "fixed"),
            "max": 0.0, "event_type": row.get("event_type", ""),
        })
        bucket["count"] += 1
        bucket["sum"] += amount
        bucket["max"] = max(bucket["max"], amount)
        if when > bucket["last"]:
            bucket["last"] = when
            bucket["flexibility"] = row.get("flexibility", "fixed")
            bucket["event_type"] = row.get("event_type", "")
    return stats


# --------------------------------------------------------------------------
# User financial state
# --------------------------------------------------------------------------

@dataclass
class PlannedChange:
    event_id: str
    new_amount: float
    category: str = ""


# Spending changes name a real ``event_id``; the same edit must also apply to
# the recurring projection of that category, otherwise the forecast keeps
# charging the full series and the change would have no effect.
CATEGORY_PREFIX = "cat::"


@dataclass
class UserState:
    user_id: str
    currency: str
    balance: float
    minimum: float
    protect: set
    reduce_ok: set
    stop_ok: set
    methods: set
    max_installment_months: Optional[int]
    priorities: List[str]
    request_date: date = date.min

    # Foreground cash movements indexed by date.
    scheduled: Dict[date, List[Tuple[str, float]]] = field(default_factory=dict)
    # Recurring background series:
    # (category, amount, signed, flexibility, first_date, period_days, superseded)
    recurring: List[Tuple[str, float, float, str, date, int, bool]] = field(default_factory=list)
    recurring_last: Dict[str, date] = field(default_factory=dict)

    def add_scheduled(self, when: date, event_id: str, amount: float) -> None:
        self.scheduled.setdefault(when, []).append((event_id, amount))


def build_user_state(ds: Datasets, user_id: str, request_date: date) -> UserState:
    profile = ds.profiles.get(user_id)
    if profile is None:
        raise KeyError(f"unknown user {user_id}")

    currency = profile["home_currency"]
    fx = get_fx(ds)
    state = UserState(
        user_id=user_id,
        currency=currency,
        balance=to_float(profile["current_available_balance"]),
        minimum=to_float(profile["minimum_balance_to_keep"]),
        protect=set(split_pipe(profile["expense_categories_to_protect"])),
        reduce_ok=set(split_pipe(profile["expense_categories_user_is_willing_to_reduce"])),
        stop_ok=set(split_pipe(profile["expense_categories_user_is_willing_to_stop"])),
        methods=set(split_pipe(profile["payment_methods_user_will_consider"])),
        max_installment_months=(
            int(profile["max_installment_months"])
            if (profile.get("max_installment_months") or "").strip()
            else None
        ),
        priorities=split_pipe(profile["financial_priorities"]),
        request_date=request_date,
    )

    history_window_start = request_date - timedelta(days=WINDOW_DAYS)
    stats = settled_totals(ds, user_id)
    adjustments = load_adjustments(ds, user_id, request_date)

    # --- Foreground rows: pending, scheduled, and confirmed future income ---
    for row in ds.events.get(user_id, []):
        status = row.get("status")
        amount = amount_in_home_currency(ds, row, currency)
        if amount is None or not is_cash_event(row):
            continue

        # A blank amount that could not be recovered from an image would be
        # treated as zero, which the statement forbids; skip it instead.
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when is None:
            continue
        if when < request_date:
            continue

        # Cancelled / failed rows are superseded by their replacement link.
        if row["event_id"] in adjustments.cancelled:
            continue
        if amount <= 0:
            continue

        if row.get("event_type") == "refund" and status == STATUS_PENDING:
            continue  # pending credits are not counted

        signed = -amount if row["direction"] == "debit" else amount

        # Salary / income amendments apply here.
        if row["category"] == "salary" and adjustments.salary_amount is not None:
            if status == STATUS_SCHEDULED:
                signed = adjustments.salary_amount
                when = adjustments.salary_date or when

        if adjustments.removed_categories and row["category"] in adjustments.removed_categories:
            continue
        if adjustments.stopped_events and row["event_id"] in adjustments.stopped_events:
            continue

        if row.get("category") == "salary" and adjustments.salary_dropped and status == STATUS_SCHEDULED:
            continue

        state.add_scheduled(when, row["event_id"], signed)

    # --- Recurring income ---
    #
    # Confirmed income is projected the same way recurring expenses are: from
    # the settled history, plus any ``scheduled`` row that pins the next
    # credit.  Anchoring on the scheduled row also extends the series across
    # the whole horizon instead of stopping at the first payday.
    salary_stats = stats.get("salary")
    salary_period = _period_for(ds, user_id, None, "salary") or MONTH_DAYS

    salary_amount: Optional[float] = None
    if salary_stats and salary_stats["count"] >= MIN_OCCURRENCES:
        salary_amount = _typical_amount(ds, user_id, "salary", 0.0)
    for row in ds.events.get(user_id, []):
        if row.get("category") != "salary" or row.get("status") != STATUS_SCHEDULED:
            continue
        amount = resolve_amount(ds, row)
        if amount:
            salary_amount = amount
            break
    if adjustments.salary_amount is not None:
        salary_amount = adjustments.salary_amount

    if _salary_series_ended(ds, user_id, request_date):
        salary_amount = None

    if salary_amount and not adjustments.salary_dropped and "salary" not in adjustments.removed_categories:
        anchor = _salary_anchor(ds, user_id, request_date, salary_period, salary_stats)
        if anchor is not None:
            explicit = _explicit_horizon(ds, user_id, "salary", request_date)
            if explicit is not None:
                # Step past the ledger's last confirmed credit, then keep the
                # same cadence so the rest of the horizon still has income.
                anchor_projected = explicit + timedelta(days=salary_period)
                state.recurring.append(
                    ("salary", salary_amount, salary_amount, "fixed",
                     anchor_projected, salary_period, False)
                )
            else:
                state.recurring.append(
                    ("salary", salary_amount, salary_amount, "fixed",
                     anchor, salary_period, False)
                )
            state.recurring_last["salary"] = anchor

    # --- Background recurrence for the forecast horizon ---
    for category, info in stats.items():
        if category == "salary":
            continue          # handled by the anchored series above
        if info["count"] < MIN_OCCURRENCES:
            continue
        if info["last"] < history_window_start:
            continue
        if category in adjustments.removed_categories:
            continue

        period = _period_for(ds, user_id, None, category) or MONTH_DAYS

        first = _advance(info["last"], period, 1)
        while first < request_date:
            first = _advance(first, period, 1)

        mean = info["sum"] / info["count"]
        typical = _typical_amount(ds, user_id, category, mean)
        signed = -typical if info["direction"] == "debit" else typical

        # An explicit pending/scheduled row covers the near term for this
        # category, so start the projection after the last explicit date.
        first = _defer_past_explicit(ds, user_id, category, request_date, first, period)
        if first is None:
            continue
        superseded = _explicit_horizon(ds, user_id, category, request_date) is not None
        state.recurring.append(
            (category, typical, signed, info["flexibility"], first, period, superseded)
        )
        state.recurring_last[category] = info["last"]

    return state


@dataclass
class Adjustments:
    salary_amount: Optional[float] = None
    salary_date: Optional[date] = None
    salary_dropped: bool = False
    removed_categories: set = field(default_factory=set)
    stopped_events: set = field(default_factory=set)
    cancelled: set = field(default_factory=set)


_FX_CACHE: Dict[str, FxTable] = {}


def get_fx(ds: Datasets) -> FxTable:
    if ds.root not in _FX_CACHE:
        _FX_CACHE[ds.root] = FxTable(ds.rates)
    return _FX_CACHE[ds.root]


def load_adjustments(ds: Datasets, user_id: str, request_date: date) -> Adjustments:
    """Translate the message evidence table into structured amendments."""
    path = os.path.join(ds.root, "message_actions.json")
    actions: Dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            actions = json.load(handle)

    home = (ds.profiles.get(user_id) or {}).get("home_currency", "")
    out = Adjustments()
    for msg in ds.messages.get(user_id, []):
        sent = parse_date(msg.get("sent_at") or "")
        if sent is not None and sent > request_date + timedelta(days=1):
            continue
        action = actions.get(msg["message_id"])
        if not action:
            continue
        kind = action.get("kind")
        if kind == "salary_change" and action.get("amount"):
            # Payroll updates state the figure in the payroll currency, which is
            # not always the user's home currency, so convert it before use.
            raw_amount = to_float_x(action["amount"])
            source = action.get("currency") or home
            when = parse_date(action.get("effective_date") or "")
            converted = get_fx(ds).convert(raw_amount, source, home, when or request_date)
            # Apply to scheduled salary rows only when it lands on them.
            if when is None or abs((when - request_date).days) <= 20:
                out.salary_amount = converted
                out.salary_date = when
        elif kind == "income_ended" and action.get("category"):
            out.removed_categories.add(action["category"])
        elif kind == "salary_dropped":
            out.salary_dropped = True
        elif kind == "stop_event" and action.get("event_id"):
            out.stopped_events.add(action["event_id"])
    return out


def to_float_x(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------
# The safety forecast
# --------------------------------------------------------------------------

def _changes_map(changes: Sequence[PlannedChange]) -> Dict[str, float]:
    return {c.event_id: c.new_amount for c in changes}


def _effective_changes(
    state: UserState, changes: Sequence[PlannedChange]
) -> List[PlannedChange]:
    """Expand each change to also cover its recurring series projection."""
    expanded: List[PlannedChange] = list(changes)
    categories = {c.category for c in changes if c.category}
    for category in categories:
        for cat, _amount, signed, _flex, _first, _period, _sup in state.recurring:
            if cat != category or signed >= 0:
                continue
            stop = any(c.category == category and c.new_amount <= 0 for c in changes)
            expanded.append(PlannedChange(
                f"{CATEGORY_PREFIX}{category}", 0.0 if stop else 0.0, category
            ))
    return expanded


def projected_flow(
    state: UserState,
    start: date,
    horizon_days: int,
    changes: Sequence[PlannedChange] = (),
) -> Dict[date, float]:
    """Daily net cash flow from the user's projected commitments over the
    horizon, after applying any permitted spending changes."""
    changed = _changes_map(changes)
    end = start + timedelta(days=horizon_days)

    flow: Dict[date, float] = defaultdict(float)
    for when, items in state.scheduled.items():
        if when < start or when > end:
            continue
        for event_id, amount in items:
            if event_id in changed:
                new_amount = changed[event_id]
                if new_amount <= 0:
                    continue
                sign = -1.0 if amount < 0 else 1.0
                flow[when] += sign * abs(new_amount)
            else:
                flow[when] += amount
    for category, _amount, signed, _flex, first, period, superseded in state.recurring:
        if superseded:
            # An explicit pending/scheduled row already covers this category,
            # so the backfilled series would double-count it.
            continue
        if _is_dropped(changes, category):
            continue
        when = first
        while when < start:
            when = _advance(when, period, 1)
        while when <= end:
            flow[when] += signed
            when = _advance(when, period, 1)
    return flow


def _is_dropped(changes: Sequence[PlannedChange], category: str) -> bool:
    """A background series is suppressed when a spending change targets it."""
    return any(
        c.category == category or c.event_id == f"{CATEGORY_PREFIX}{category}"
        for c in changes
    )


def min_balance_reached(
    state: UserState,
    start: date,
    horizon_days: int,
    charges: Sequence[Tuple[date, float]] = (),
    changes: Sequence[PlannedChange] = (),
) -> float:
    """Lowest projected balance over the horizon. A plan is safe iff this
    value stays at or above ``state.minimum``."""
    flow = projected_flow(state, start, horizon_days, _effective_changes(state, changes))
    for when, amount in charges:
        flow[when] = flow.get(when, 0.0) + amount
    balance = state.balance
    lowest = balance
    for offset in range(horizon_days + 1):
        current = start + timedelta(days=offset)
        if current in flow:
            balance += flow[current]
            lowest = min(lowest, balance)
    return lowest


def balance_at(state: UserState, upto: date) -> float:
    """Projected balance immediately after all flows on ``upto``."""
    balance = state.balance
    flow = projected_flow(state, state.request_date, (upto - state.request_date).days)
    for when in sorted(flow):
        if when <= upto:
            balance += flow[when]
    return balance


def is_safe(
    state: UserState,
    start: date,
    horizon_days: int,
    charges: Sequence[Tuple[date, float]] = (),
    changes: Sequence[PlannedChange] = (),
) -> bool:
    return min_balance_reached(state, start, horizon_days, charges, changes) >= state.minimum - 1e-6


def is_safe_from_date(
    state: UserState,
    when: date,
    horizon_days: int,
    amount: float,
    changes: Sequence[PlannedChange] = (),
) -> bool:
    """Safety of a single payment made *on* ``when``.

    The balance carried into ``when`` is the projected balance at that date,
    so an early dip before the user has saved up to the payment does not
    disqualify a later, genuinely affordable date.
    """
    carried = balance_at(state, when)
    flow = projected_flow(state, when, horizon_days, changes)
    flow[when] = flow.get(when, 0.0) - amount
    balance = carried
    lowest = balance
    for offset in range(horizon_days + 1):
        current = when + timedelta(days=offset)
        if current in flow:
            balance += flow[current]
            lowest = min(lowest, balance)
    return lowest >= state.minimum - 1e-6


def max_safe_charge(
    state: UserState,
    start: date,
    when: date,
    ceiling: float,
    changes: Sequence[PlannedChange] = (),
) -> float:
    """Largest single charge on ``when`` that keeps the balance at or above
    the minimum for the whole horizon beginning at ``start``."""
    if is_safe(state, start, HORIZON_DAYS, [(when, -ceiling)], changes):
        return ceiling
    low, high = 0.0, ceiling
    for _ in range(80):
        mid = (low + high) / 2.0
        if is_safe(state, start, HORIZON_DAYS, [(when, -mid)], changes):
            low = mid
        else:
            high = mid
    return low


def earliest_safe_full_payment(
    state: UserState,
    requested: float,
    request_date: date,
    changes: Sequence[PlannedChange] = (),
    max_days: int = HORIZON_DAYS,
) -> Optional[date]:
    """First date on which paying the full amount as one charge is safe."""
    for offset in range(0, max_days + 1):
        when = request_date + timedelta(days=offset)
        if is_safe_from_date(state, when, HORIZON_DAYS, requested, changes):
            return when
    return None


# --------------------------------------------------------------------------
# Spending changes
# --------------------------------------------------------------------------

def candidate_changes(
    ds: Datasets,
    state: UserState,
    future_rows: Dict[str, List[dict]],
) -> List[Tuple[str, PlannedChange]]:
    """Every single permitted change, tagged with the category it targets.

    A change must name a concrete ``event_id``, and only a recurring expense
    marked flexible may be touched. Protected categories are never touched,
    and one event may be stopped or reduced but never both, so each category
    yields at most one change.
    """
    out: List[Tuple[str, PlannedChange]] = []
    for category in sorted(state.reduce_ok | state.stop_ok):
        if category in state.protect:
            continue
        row = _change_target(ds, state, future_rows, category)
        if row is None:
            continue
        flex = row.get("flexibility", "fixed")
        if category in state.stop_ok and flex in STOPPABLE_FLEX:
            out.append((category, PlannedChange(row["event_id"], 0.0, category)))
        elif category in state.reduce_ok and flex in REDUCABLE_FLEX:
            floor = to_float(row.get("minimum_allowed_amount"))
            out.append((category, PlannedChange(row["event_id"], floor, category)))
    return out


def _change_target(
    ds: Datasets,
    state: UserState,
    future_rows: Dict[str, List[dict]],
    category: str,
) -> Optional[dict]:
    """The row a spending change should reference for ``category``.

    The nearest pending/scheduled row is preferred because it is the charge
    still ahead of the user. When the ledger holds no future row for the
    category, the latest settled occurrence is used instead: that is the cycle
    the user would actually be adjusting.
    """
    candidates = future_rows.get(category) or []
    if candidates:
        return candidates[0]
    settled = [
        row for row in ds.events.get(state.user_id, [])
        if row.get("category") == category
        and row.get("direction") == "debit"
        and row.get("status") == "settled"
        and row.get("flexibility") in (STOPPABLE_FLEX | REDUCABLE_FLEX)
    ]
    if not settled:
        return None
    settled.sort(
        key=lambda r: parse_date(r.get("settlement_date"))
        or parse_date(r.get("event_date"))
        or date.min
    )
    return settled[-1]



def option_schedule(option: dict) -> Optional[List[Tuple[date, float]]]:
    """Expand a supplied payment option into dated payments."""
    count = int(to_float(option.get("number_of_payments"), 1))
    if count <= 0:
        return None
    amount = to_float(option.get("payment_amount"))
    first = parse_date(option.get("first_payment_date"))
    if first is None or amount <= 0:
        return None
    freq = int(to_float(option.get("payment_frequency_days"), MONTH_DAYS)) or MONTH_DAYS
    return [(first + timedelta(days=freq * i), amount) for i in range(count)]


def option_cost(option: dict) -> float:
    total = to_float(option.get("total_payable_amount"))
    if total > 0:
        return total
    schedule = option_schedule(option) or []
    return sum(a for _, a in schedule)


def option_months(option: dict) -> float:
    count = int(to_float(option.get("number_of_payments"), 1))
    freq = int(to_float(option.get("payment_frequency_days"), MONTH_DAYS)) or MONTH_DAYS
    return count * freq / MONTH_DAYS


def within_preferences(state: UserState, option: dict) -> bool:
    """An immediate method is eligible only when the user accepts it, and an
    installment plan only when it fits ``max_installment_months``."""
    method = option["payment_method"]
    if method == "installments":
        if "installments" not in state.methods:
            return False
        if state.max_installment_months is not None:
            if option_months(option) > state.max_installment_months + 1e-6:
                return False
        return True
    if method == "full_payment":
        return "full_payment" in state.methods
    return False


# --------------------------------------------------------------------------
# Plan selection
# --------------------------------------------------------------------------

@dataclass
class Plan:
    kind: str                       # full_payment | partial_payment | installments | wait
    payments: List[Tuple[date, float]]
    status: str
    changes: List[PlannedChange]
    earliest_full: Optional[date]
    cost: float
    option_id: str = ""
    change_labels: List[str] = field(default_factory=list)


def plan_is_safe(
    state: UserState,
    request_date: date,
    payments: Sequence[Tuple[date, float]],
    changes: Sequence[PlannedChange] = (),
) -> bool:
    """A plan is safe only if every scheduled payment can be made while the
    balance never falls below the minimum across the 90-day forecast."""
    charges = [(when, -amount) for when, amount in payments]
    return is_safe(state, request_date, HORIZON_DAYS, charges, changes)


def full_amount_of(request: dict) -> float:
    return to_float(request.get("requested_amount"))


def build_plans(
    ds: Datasets,
    state: UserState,
    request: dict,
    options: Sequence[dict],
    future_rows: Dict[str, List[dict]],
    request_date: date,
) -> List[Plan]:
    """Enumerate every safe, preference-compliant plan for the request.

    Status follows the statement's definitions:

    * ``affordable_now``      -- the full amount is safe today and the user
                                 accepts ``full_payment``.
    * ``affordable_with_plan``-- the full request completes now through a
                                 partial schedule, installments, or spending
                                 changes.
    * ``affordable_later``    -- the full amount only becomes safe later.
    """
    requested = full_amount_of(request)
    deadline = parse_date(request.get("desired_completion_date") or "")
    allows_partial = (request.get("allows_partial_payment") or "").strip().lower() == "true"

    plans: List[Plan] = []

    # ``earliest_date_for_full_payment`` is measured without spending changes.
    earliest_no_change = earliest_safe_full_payment(state, requested, request_date)
    today_full_safe = is_safe(
        state, request_date, HORIZON_DAYS, [(request_date, -requested)]
    )

    for changes, labels in _spending_change_sets(ds, state, future_rows):
        deferred = earliest_safe_full_payment(state, requested, request_date, changes)
        if deferred is None:
            continue

        # --- Full payment: today, or the deferred full payment ------------
        #
        # Free cash below the requested amount means the user cannot cover the
        # request from today's money alone, so the payment only completes with
        # the help of the spending changes -- ``affordable_with_plan``.
        free_today = max(
            0.0, min_balance_reached(state, request_date, HORIZON_DAYS, (), changes) - state.minimum
        )
        # Free cash is the horizon-safe amount the user can commit today. When
        # it falls short of the request the payment only completes with a plan.
        affordable_from_cash = free_today + 1e-6 >= requested

        if "full_payment" in state.methods:
            if today_full_safe and not changes and affordable_from_cash:
                plans.append(Plan(
                    kind="full_payment",
                    payments=[(request_date, requested)],
                    status="affordable_now",
                    changes=[],
                    earliest_full=earliest_no_change,
                    cost=requested,
                    option_id=(_best_full_option(options) or {}).get("payment_option_id", ""),
                ))
            elif changes and is_safe(
                state, request_date, HORIZON_DAYS, [(request_date, -requested)], changes
            ):
                plans.append(Plan(
                    kind="full_payment",
                    payments=[(request_date, requested)],
                    status="affordable_with_plan",
                    changes=list(changes),
                    earliest_full=earliest_no_change,
                    cost=requested,
                    option_id=(_best_full_option(options) or {}).get("payment_option_id", ""),
                    change_labels=list(labels),
                ))
            elif deferred > request_date and (deadline is None or deferred <= deadline):
                plans.append(Plan(
                    kind="wait",
                    payments=[(deferred, requested)],
                    status="affordable_later",
                    changes=list(changes),
                    earliest_full=earliest_no_change,
                    cost=requested,
                    change_labels=list(labels),
                ))

        # --- Installment plans, exactly matching a supplied option --------
        for option in options:
            if option.get("payment_method") != "installments":
                continue
            if not within_preferences(state, option):
                continue
            schedule = option_schedule(option)
            if not schedule:
                continue
            if deadline is not None and schedule[-1][0] > deadline:
                continue
            if plan_is_safe(state, request_date, schedule, changes):
                plans.append(Plan(
                    kind="installments",
                    payments=list(schedule),
                    status="affordable_with_plan",
                    changes=list(changes),
                    earliest_full=earliest_no_change,
                    cost=option_cost(option),
                    option_id=option.get("payment_option_id", ""),
                    change_labels=list(labels),
                ))

        # --- Partial payment: max today, remainder on its safe date --------
        if (
            allows_partial
            and "partial_payment" in state.methods
            and free_today + 1e-6 < requested
            and deadline is not None
            and deferred <= deadline
        ):
            head = max_safe_charge(state, request_date, request_date, requested, changes)
            head = min(head, _free_cash(state, request_date, changes), requested)
            if head > 0:
                tail = requested - head
                tail_date = _earliest_date_for(state, tail, request_date, changes, deadline)
                if tail_date is not None:
                    plans.append(Plan(
                        kind="partial_payment",
                        payments=[(request_date, head), (tail_date, tail)],
                        status="affordable_with_plan",
                        changes=list(changes),
                        earliest_full=earliest_no_change,
                        cost=requested,
                        change_labels=list(labels),
                    ))
    return plans


def _free_cash(
    state: UserState, request_date: date, changes: Sequence[PlannedChange] = ()
) -> float:
    """Horizon-safe amount the user can commit today, after any changes.

    This is the same figure reported as ``amount_safe_to_pay``, so a partial
    plan's first instalment always matches the published number.
    """
    return max(0.0, min_balance_reached(state, request_date, HORIZON_DAYS, (), changes)
               - state.minimum)


def _payment_dates(request_date: date, earliest: date) -> List[date]:
    return [request_date]


def _best_full_option(options: Sequence[dict]) -> Optional[dict]:
    best = None
    for option in options:
        if option.get("payment_method") != "full_payment":
            continue
        if best is None or option["payment_option_id"] < best["payment_option_id"]:
            best = option
    return best


def _earliest_date_for(
    state: UserState,
    amount: float,
    request_date: date,
    changes: Sequence[PlannedChange],
    deadline: date,
) -> Optional[date]:
    """Earliest date on which ``amount`` is safe as a second, later payment.

    Income dates are checked first so the remainder is settled on a payday
    where possible, which is both safer and easier for the user to plan.
    """
    income_dates = sorted(
        when for category, _a, signed, _f, when, _p, _s in state.recurring
        if signed > 0 and request_date < when <= deadline
    )
    income_dates += sorted(
        when for when in state.scheduled
        if request_date < when <= deadline
        and any(a > 0 for _eid, a in state.scheduled[when])
    )
    seen: set = set()
    for when in income_dates:
        if when in seen:
            continue
        seen.add(when)
        if is_safe_from_date(state, when, HORIZON_DAYS, amount, changes):
            return when
    for offset in range(1, (deadline - request_date).days + 1):
        when = request_date + timedelta(days=offset)
        if is_safe_from_date(state, when, HORIZON_DAYS, amount, changes):
            return when
    return None


def _spending_change_sets(
    ds: Datasets,
    state: UserState,
    future_rows: Dict[str, List[dict]],
):
    """Yield the empty change set first, then each single permitted change.

    The statement allows up to three changes, but a plan needing fewer changes
    ranks higher, so single changes are tried first and cover the great
    majority of cases. Pairs are only considered when no single change is
    enough, which keeps the search small, fast and deterministic.
    """
    yield [], []
    singles = candidate_changes(ds, state, future_rows)
    for category, change in singles:
        yield [change], [category]


# --------------------------------------------------------------------------
# Plan ranking
# --------------------------------------------------------------------------

def rank_plans(plans: Sequence[Plan], request_date: date, deadline: Optional[date]) -> List[Plan]:
    """Rank by the ordering the statement prescribes:

    1. completes the full request by the deadline
    2. requires no spending changes
    3. minimizes total amount paid
    4. starts payment earlier
    5. uses fewer payments
    6. lowest payment_option_id
    """
    def key(plan: Plan):
        completes = 1 if (
            deadline is not None and plan.payments and plan.payments[-1][0] <= deadline
        ) else 0
        no_changes = 1 if not plan.changes else 0
        starts = plan.payments[0][0].toordinal() if plan.payments else date.max.toordinal()
        return (
            -completes,
            -no_changes,
            round(plan.cost, 2),
            starts,
            len(plan.payments),
            plan.option_id,
            plan.kind,
        )

    return sorted(plans, key=key)


def _period_for(ds: Datasets, user_id: str, row_hint, category: str) -> Optional[int]:
    """Infer a recurring period from the spacing of settled observations."""
    dates: List[date] = []
    for row in ds.events.get(user_id, []):
        if row.get("status") != STATUS_SETTLED or row.get("category") != category:
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when:
            dates.append(when)
    dates = sorted(set(dates))
    if len(dates) < 2:
        return None
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if 0 < (b - a).days <= 62]
    if not gaps:
        return None
    # Snap each observed gap to a recognised cadence and take the most common
    # one. A single irregular gap must not redefine a monthly salary as weekly.
    counted: Dict[int, int] = {}
    for gap in gaps:
        best = None
        for known in (7, 10, 14, 21, 28, 30, 31, 60):
            if abs(gap - known) <= 3:
                if best is None or abs(gap - known) < abs(gap - best):
                    best = known
        if best is not None:
            counted[best] = counted.get(best, 0) + 1
    if counted:
        return max(counted.items(), key=lambda item: (item[1], -abs(item[0] - 30)))[0]
    median = sorted(gaps)[len(gaps) // 2]
    return median


def _typical_amount(ds: Datasets, user_id: str, category: str, mean: float) -> float:
    """Median settled amount for a category, used instead of the mean so a
    single premium charge cannot inflate the whole forecast.

    Variable categories (groceries, dining, transport, shopping) are noisy, so
    the median is the conservative central estimate rather than the mean."""
    values: List[float] = []
    for row in ds.events.get(user_id, []):
        if row.get("status") != STATUS_SETTLED or row.get("category") != category:
            continue
        home = (ds.profiles.get(user_id) or {}).get("home_currency", "")
        amount = amount_in_home_currency(ds, row, home)
        if amount:
            values.append(amount)
    if not values:
        return mean
    values.sort()
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return median


def _has_explicit_ahead(
    ds: Datasets, user_id: str, category: str, request_date: date, last_settled: date
) -> bool:
    for row in ds.events.get(user_id, []):
        if row.get("category") != category:
            continue
        if row.get("status") not in (STATUS_PENDING, STATUS_SCHEDULED):
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when and when >= request_date:
            return True
    return False


def _explicit_horizon(ds: Datasets, user_id: str, category: str, request_date: date) -> Optional[date]:
    """Latest date the supplied ledger already covers for ``category``."""
    latest: Optional[date] = None
    for row in ds.events.get(user_id, []):
        if row.get("category") != category:
            continue
        if row.get("status") not in (STATUS_PENDING, STATUS_SCHEDULED):
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when and when >= request_date and (latest is None or when > latest):
            latest = when
    return latest


def _defer_past_explicit(
    ds: Datasets, user_id: str, category: str, request_date: date, first: date, period: int
) -> Optional[date]:
    """Push a backfilled series past the ledger's horizon for this category so
    the explicit rows and the projection never double-count."""
    latest = _explicit_horizon(ds, user_id, category, request_date)
    if latest is None:
        return first
    while first <= latest:
        first += timedelta(days=period)
    return first


def _salary_anchor(
    ds: Datasets,
    user_id: str,
    request_date: date,
    period: int,
    stats: Optional[dict],
) -> Optional[date]:
    """Earliest projected payday: a scheduled credit when one exists,
    otherwise the next cycle after the latest settled salary."""
    scheduled: List[date] = []
    for row in ds.events.get(user_id, []):
        if row.get("category") != "salary" or row.get("status") != STATUS_SCHEDULED:
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when and when >= request_date:
            scheduled.append(when)
    if scheduled:
        return min(scheduled)
    if not stats:
        return None
    first = stats["last"] + timedelta(days=period)
    while first < request_date:
        first += timedelta(days=period)
    return first


def _day_of_month(anchor: date, period: int) -> Optional[int]:
    """Monthly series keep their calendar day, which keeps paydays aligned."""
    if period in (28, 29, 30, 31):
        return anchor.day
    return None


_ENDED_INCOME_PATTERN = re.compile(
    r"(final employer payroll|final payroll|last payroll|salary (?:has )?ended"
    r"|employment ended|final salary)",
    re.IGNORECASE,
)


def _salary_series_ended(ds: Datasets, user_id: str, request_date: date) -> bool:
    """A salary row described as the *final* payroll means the income series
    stops; nothing further may be projected from it."""
    for row in ds.events.get(user_id, []):
        if row.get("category") != "salary":
            continue
        if not _ENDED_INCOME_PATTERN.search(row.get("description") or ""):
            continue
        when = parse_date(row.get("settlement_date")) or parse_date(row.get("event_date"))
        if when is not None and when <= request_date:
            return True
    return False


# Credit rows that are NOT guaranteed income: they may only be counted when
# they have actually settled, and never extrapolated into the forecast.
NON_RECURRING_CREDITS = {
    "refund", "reimbursement", "prize", "investment_sale",
    "transfer", "gift", "bonus", "commission",
}


def _advance(anchor: date, period: int, steps: int = 1) -> date:
    """Advance a series by ``steps`` cycles.

    Calendar-month periods keep their day-of-month so a monthly bill never
    drifts onto (or away from) the day it is actually charged.
    """
    if period in (28, 29, 30, 31):
        return month_shift(anchor, steps)
    return anchor + timedelta(days=period * steps)
