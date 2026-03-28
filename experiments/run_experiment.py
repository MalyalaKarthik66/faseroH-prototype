import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end experiment runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    dataset_path = Path(ds["path"])

    if not args.skip_data or not dataset_path.exists():
        run(
            [
            sys.executable,
                "data/generate_taylor_dataset.py",
                "--size",
                str(ds.get("size", 20000)),
                "--max-order",
                str(ds.get("max_order", 4)),
                "--seed",
                str(cfg.get("seed", 42)),
                "--output",
                str(dataset_path),
            ]
        )
        run([sys.executable, "data/validate_dataset.py", "--input", str(dataset_path)])

    run([
        sys.executable,
        "training/train.py",
        "--config",
        args.config,
        "--results-dir",
        args.results_dir,
    ])
    print("Experiment completed. See results/ for artifacts.")


if __name__ == "__main__":
    main()
