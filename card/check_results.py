"""Check that make_card.py emits exactly the numbers on the model card.

    python card/make_card.py              # regenerate card/results.json
    python card/check_results.py          # compare it with card/results.sha256
    python card/check_results.py --write  # record a new digest after an intended change

The JSON is hashed after rounding every float to ten significant digits, finer than any number the card prints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE / "results.sha256"
OUTPUTS = ("results.json",)
DIGITS = 10


def rounded(x):
    if isinstance(x, float):
        return float(f"{x:.{DIGITS}g}")
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    return x


def digest(path: Path) -> str:
    return hashlib.sha256(json.dumps(rounded(json.loads(path.read_text())), separators=(",", ":")).encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="record the current digest in results.sha256")
    args = ap.parse_args(argv)
    now = {name: digest(HERE / name) for name in OUTPUTS if (HERE / name).exists()}
    if not now:
        print("no outputs found: run python card/make_card.py first", file=sys.stderr)
        return 1
    if args.write:
        RECORD.write_text("".join(f"{h}  {name}\n" for name, h in now.items()))
        print(f"recorded {len(now)} digest(s) in {RECORD.name}")
        return 0
    expected = {name: h for h, name in (line.split(maxsplit=1) for line in RECORD.read_text().splitlines() if line.strip())}
    bad = [name for name in expected if now.get(name) != expected[name]]
    for name in bad:
        print(f"MISMATCH {name}", file=sys.stderr)
    print("card results match" if not bad else f"{len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
