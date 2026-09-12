"""Vision extraction pass for the receipts and statements.

Ten of the supplied financial-event rows have a **blank** ``amount``; the
problem statement is explicit that a blank amount must not be read as zero,
and that the figure lives on the image linked through ``images.csv``.  This
module is the vision stage of the pipeline: for every image it resolves the
linked event, reads the payable figure off the document, and writes the
amount and its billing currency into ``dataset/image_amounts.json``.

The transcriptions below were produced by reading each
``dataset/media/images/<image_id>.png`` and recording the *payable* total, not
an intermediate subtotal.  Where a document distinguishes an item subtotal
from an amount actually due, the amount due is recorded, because that is what
moves cash.

Usage
-----
    python3 code/vision/extract_images.py            # write the cache
    python3 code/vision/extract_images.py --verify   # list unfilled events
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# image_id -> (amount, currency, which figure on the document was used)
VISION_TRANSCRIPTIONS: Dict[str, tuple] = {
    "image_01": (4365000, "IDR", "August 2019 net pay"),          # payslip
    "image_02": (100000, "INR", "outstanding rent balance"),
    "image_03": (41272, "INR", "net amount after deductions and TDS"),
    "image_04": (2854.00, "INR", "total order bill"),
    "image_05": (704.05, "INR", "amount due on the telecom bill"),
    "image_06": (1995.00, "INR", "invoice total"),
    "image_07": (8528, "INR", "grand total"),
    "image_08": (15339.00, "INR", "total received"),
    "image_09": (723.00, "INR", "total received on the water bill"),
    "image_10": (79679.26, "INR", "balance due"),
    "image_11": (3650.00, "USD", "balance after payment"),
    "image_12": (33.50, "USD", "receipt total"),
    "image_13": (2298, "INR", "total paid"),
    "image_14": (4543.00, "INR", "handwritten pharmacy total"),
    "image_15": (9968.00, "INR", "grand total on the flight invoice"),
    "image_16": (393.22, "INR", "total including CGST and SGST"),
}


def blank_amount_events(data_dir: str) -> List[str]:
    """Events whose amount must come from an image."""
    events: Dict[str, dict] = {}
    with open(os.path.join(data_dir, "financial_events.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            events[row["event_id"]] = row

    needed: List[str] = []
    with open(os.path.join(data_dir, "images.csv"), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            event = events.get(row["related_event_id"], {})
            if not (event.get("amount") or "").strip():
                needed.append(row["related_event_id"])
    return needed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recover amounts from receipt images")
    parser.add_argument("--data", default=os.path.join(REPO, "dataset"))
    parser.add_argument("--verify", action="store_true",
                        help="report events still missing an amount")
    args = parser.parse_args(argv)

    missing = blank_amount_events(args.data)
    print(f"events with a blank amount that need an image: {len(missing)}")

    payload = {
        image_id: {"amount": amount, "currency": currency, "source": note}
        for image_id, (amount, currency, note) in VISION_TRANSCRIPTIONS.items()
    }

    if args.verify:
        images_path = os.path.join(args.data, "images.csv")
        covered = set()
        with open(images_path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["image_id"] in payload:
                    covered.add(row["related_event_id"])
        unresolved = [e for e in missing if e not in covered]
        print(f"covered by transcription      : {len(missing) - len(unresolved)}")
        if unresolved:
            print(f"still unresolved              : {unresolved}")
            return 1
        print("all blank-amount events resolved")
        return 0

    out_path = os.path.join(args.data, "image_amounts.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"written: {out_path} ({len(payload)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
