"""Command-line entry point for the reproducible audit."""

import argparse
import json

from .study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="out", help="Directory for generated audit files")
    parser.add_argument(
        "--n", type=int, default=39, help="Interior points per side (default: 39)"
    )
    parser.add_argument("--rank", type=int, default=20, help="Retained POD modes (default: 20)")
    args = parser.parse_args(argv)
    try:
        summary = run_study(args.output, args.n, args.rank)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
