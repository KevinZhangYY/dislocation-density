from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dislo_density")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run dislocation density estimation")
    run_p.add_argument("--input", required=True, help="Input image file or folder")
    run_p.add_argument("--out", required=True, help="Output folder")
    run_p.add_argument("--config", required=True, help="YAML config path")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        cfg = load_config(Path(args.config))
        run_batch(
            input_path=Path(args.input),
            out_dir=Path(args.out),
            cfg=cfg,
        )
        return 0

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
