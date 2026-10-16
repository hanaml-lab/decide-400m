"""Check that results/summary.json is what scripts/analyze.py computes from the committed predictions.

    python scripts/analyze.py --results results --out /tmp/summary.json
    python scripts/check_summary.py /tmp/summary.json results/summary.json

Every number is compared with a relative tolerance of 1e-6 (and an absolute one of 1e-9), so the check holds across
BLAS builds but catches any changed result.
"""
from __future__ import annotations

import json
import math
import sys


def walk(a, b, path, bad):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                bad.append(f"{path}/{k}: only in {'new' if k in a else 'recorded'}")
            else:
                walk(a[k], b[k], f"{path}/{k}", bad)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", bad)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        # NaN matches NaN: the asked-alone check of the model without the none slot compares its -inf slot logits
        if not (math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) or (math.isinf(a) and a == b) or (math.isnan(a) and math.isnan(b))):
            bad.append(f"{path}: {a} != {b}")
    elif a != b:
        bad.append(f"{path}: {a!r} != {b!r}")


def main(argv=None) -> int:
    new, recorded = (argv or sys.argv[1:])[:2]
    bad: list[str] = []
    walk(json.load(open(new)), json.load(open(recorded)), "", bad)
    for line in bad[:50]:
        print("MISMATCH", line, file=sys.stderr)
    print("summary matches" if not bad else f"{len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
