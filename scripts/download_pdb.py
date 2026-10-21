from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a PDB structure from RCSB.")
    parser.add_argument("pdb_id", help="Four-character PDB identifier, e.g. 1O01")
    parser.add_argument("--output", type=Path, default=Path("data/01_raw_receptors/receptor.pdb"))
    args = parser.parse_args()
    pdb_id = args.pdb_id.strip().upper()
    if len(pdb_id) != 4:
        raise SystemExit("PDB ID must contain exactly four characters.")
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(response.text, encoding="utf-8")
    print(f"Saved {pdb_id} to {args.output}")


if __name__ == "__main__":
    main()
