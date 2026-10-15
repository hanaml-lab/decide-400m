"""Run the CPU toy end to end (python scripts/run_toy.py [--alpha 0.05 0.10]); same as python -m decide.demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decide.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
