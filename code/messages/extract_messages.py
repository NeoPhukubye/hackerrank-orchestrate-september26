"""Message understanding pass.

``dataset/messages.csv`` carries the unstructured half of the financial
picture: payroll amendments, income that has ended, refunds that have not
landed, cancellations, and prize/loan offers that must never be treated as
cash.  This module turns those sentences into the structured actions that
``code/engine.py`` consumes from ``dataset/message_actions.json``.

The extractor is pattern based so the run stays deterministic and cheap; it
performs no arithmetic and never invents a monetary fact.  Message text is
treated strictly as untrusted evidence: an instruction embedded in a message
cannot change the challenge rules, only the financial facts it describes.

Usage
-----
    python3 code/messages/extract_messages.py            # writes the JSON cache
    python3 code/messages/extract_messages.py --print    # show what was found
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --------------------------------------------------------------------------
# Patterns.  Each one targets a distinct financial fact, and several are
# written twice so both the English and Indonesian wordings are recognised.
# --------------------------------------------------------------------------

SALARY_CHANGE = [
    re.compile(r"salary (?:has )?(?:increased|risen) to ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"monthly salary has increased to ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"gaji bulanan anda naik menjadi ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(
        r"(?:base salary|regular salary) (?:for the next payroll )?is ([A-Z]{3})? ?([\d,\.]+)",
        re.I,
    ),
    re.compile(r"gaji pokok yang dikonfirmasi adalah ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"next salary is reduced to ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"temporary monthly pay is ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"gaji bulanan sementara anda adalah ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"salary of ([A-Z]{3})? ?([\d,\.]+) is confirmed for ([\d-]{10})", re.I),
    re.compile(r"first salary will be ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"gaji pertama (?:anda sebesar |dari perusahaan baru adalah )([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"salary of ([A-Z]{3})? ?([\d,\.]+) resumes on ([\d-]{10})", re.I),
    re.compile(r"confirmed monthly salary is ([A-Z]{3})? ?([\d,\.]+)", re.I),
    re.compile(r"sisa gaji bulanan yang dikonfirmasi adalah ([A-Z]{3})? ?([\d,\.]+)", re.I),
]

DATE_IN_TEXT = re.compile(r"(\d{4}-\d{2}-\d{2})")

INCOME_ENDED = [
    re.compile(r"quarterly bonus is still", re.I),
    re.compile(r"employment has ended", re.I),
    re.compile(r"hubungan kerja anda telah berakhir", re.I),
    re.compile(r"seasonal contract has ended", re.I),
    re.compile(r"kontrak musiman saat ini telah berakhir", re.I),
    re.compile(r"commission shown for open deals is still pending", re.I),
    re.compile(r"komisi dari transaksi yang masih berjalan belum disetujui", re.I),
    re.compile(r"one household employment record has ended", re.I),
    re.compile(r"salah satu sumber pendapatan kerja rumah tangga telah berakhir", re.I),
    re.compile(r"no off-season income or renewal has been confirmed", re.I),
    re.compile(r"belum ada pendapatan di luar musim", re.I),
]

SALARY_DROPPED = [
    re.compile(r"no regular salary payments scheduled after the final settlement", re.I),
    re.compile(r"tidak ada pembayaran gaji rutin yang dijadwalkan", re.I),
]

# Credits that are explicitly *not* available cash yet.
UNSETTLED_CREDIT = [
    re.compile(r"payout is still pending", re.I),
    re.compile(r"pembayaran .* masih tertunda", re.I),
    re.compile(r"prize claim has been verified and is still in payment processing", re.I),
    re.compile(r"refund has been initiated but has not reached your account", re.I),
    re.compile(r"dispute is open and no reversal has been posted", re.I),
    re.compile(r"bill is still outstanding and another debit", re.I),
]

# Credits that are recoverable but must not be extrapolated.
SETTLED_CREDIT = [
    re.compile(r"prize proceeds have reached your account", re.I),
    re.compile(r"hasil penjualan investasi anda sudah masuk", re.I),
    re.compile(r"proceeds from your investment sale have settled", re.I),
]

NON_CASH = [
    re.compile(r"no units have been sold and no cash proceeds", re.I),
    re.compile(r"matching debit and credit came from a transfer between your two accounts", re.I),
    re.compile(r"displayed market value has increased", re.I),
    re.compile(r"pay the release charge today", re.I),
    re.compile(r"prize claim", re.I),
]

REFUND_PENDING = re.compile(r"refund has been initiated but has not reached", re.I)


def normalise_number(raw: str) -> Optional[float]:
    """Parse an amount that may use either 1,234.56 or 1.234.567 grouping."""
    # Trailing sentence punctuation is not part of the figure.
    raw = raw.strip().rstrip(".,;:").strip()
    if not raw:
        return None
    if "," in raw and "." in raw:
        # The last separator is the decimal point.
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        head, _, tail = raw.rpartition(",")
        raw = raw.replace(",", "") if len(tail) == 3 else raw.replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def classify(text: str) -> Optional[dict]:
    """Return the structured action a message implies, or ``None``.

    Order matters. A payroll update can mention a pending commission in the
    same breath as a *confirmed* salary, and the confirmed figure is the fact
    that matters, so salary patterns are evaluated before the "income ended"
    and "credit not settled" heuristics.
    """
    for pattern in SALARY_CHANGE:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        currency = None
        number = None
        for group in groups:
            if group and re.fullmatch(r"[A-Z]{3}", group):
                currency = group
            elif group and re.match(r"^[\d,\.]+$", group):
                number = group
        amount = normalise_number(number) if number else None
        if amount:
            action = {"kind": "salary_change", "amount": amount, "currency": currency}
            dates = DATE_IN_TEXT.findall(text)
            for candidate in dates:
                action["effective_date"] = candidate
            return action

    for pattern in NON_CASH:
        if pattern.search(text):
            if "transfer between your two accounts" in text or \
                    "two accounts" in text:
                return {"kind": "internal_transfer"}
            if "displayed market value" in text:
                return {"kind": "unrealized_investment"}
            return {"kind": "no_action"}

    for pattern in SALARY_DROPPED:
        if pattern.search(text):
            return {"kind": "salary_dropped"}

    for pattern in INCOME_ENDED:
        if pattern.search(text):
            # "Commission still pending" alongside a confirmed salary is not the
            # income ending; only the unambiguous endings are treated as such.
            if "salary" in text.lower() and re.search(
                r"(confirmed|adalah|is|resumes)\s", text, re.I
            ) and not re.search(r"(has ended|telah berakhir|no off-season)", text, re.I):
                continue
            return {"kind": "income_ended", "category": "salary"}

    if REFUND_PENDING.search(text) or any(p.search(text) for p in UNSETTLED_CREDIT):
        return {"kind": "unsettled_credit"}

    for pattern in SETTLED_CREDIT:
        if pattern.search(text):
            return {"kind": "settled_credit"}

    return None


def extract(messages_path: str) -> Dict[str, dict]:
    actions: Dict[str, dict] = {}
    with open(messages_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            action = classify(row["message_text"])
            if action:
                actions[row["message_id"]] = action
    return actions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract financial actions from messages")
    parser.add_argument("--data", default=os.path.join(REPO, "dataset"))
    parser.add_argument("--print", dest="show", action="store_true")
    args = parser.parse_args(argv)

    messages_path = os.path.join(args.data, "messages.csv")
    actions = extract(messages_path)

    if args.show:
        for message_id, action in sorted(actions.items()):
            print(f"{message_id}: {json.dumps(action, ensure_ascii=False)}")

    out_path = os.path.join(args.data, "message_actions.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(actions, handle, indent=2, ensure_ascii=False, sort_keys=True)

    counts: Dict[str, int] = {}
    for action in actions.values():
        counts[action["kind"]] = counts.get(action["kind"], 0) + 1
    print(f"messages processed: {len(actions)} actions -> {out_path}")
    for kind, count in sorted(counts.items()):
        print(f"  {kind:24} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
