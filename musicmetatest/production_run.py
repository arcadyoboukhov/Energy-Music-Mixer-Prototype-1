"""production_run.py
Simple runner that computes production features and optionally merges them
into an existing features JSON (e.g., results from `audio_features.py`).
"""

import argparse
import json
import os
import logging

from typing import Any

import production


def main():
    parser = argparse.ArgumentParser(description="Run production/mixing analysis and merge with features JSON")
    parser.add_argument("input", help="input audio file path")
    parser.add_argument("--out-json", default="features_all_with_production.json", help="output JSON file")
    parser.add_argument("--merge", default=None, help="existing features JSON to merge into")
    parser.add_argument("--target-sr", type=int, default=None, help="optional resample SR for production analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Computing production features for %s", args.input)
    prod = production.compute_production_features_from_file(args.input, target_sr=args.target_sr)

    base: dict[str, Any] = {}
    if args.merge and os.path.exists(args.merge):
        try:
            with open(args.merge, "r", encoding="utf-8") as fh:
                base = json.load(fh)
        except Exception:
            logging.warning("Failed to load merge file %s — starting with empty base", args.merge)

    base["production"] = prod

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2)

    logging.info("Wrote production features to %s", args.out_json)


if __name__ == "__main__":
    main()
