"""Command line interface: generate / run / baseline / gate / adapters."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .adapters import BUILTIN, AdapterError, load_adapter
from .gate import GateError, compare
from .generator import generate
from .models import GeneratorConfig, RunResult
from .report import render_gate, render_run
from .runner import run_adapter

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR = 0, 1, 2


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=int(_env("MEMORYGATE_SEED", "7")))
    p.add_argument("--customers", type=int, default=int(_env("MEMORYGATE_CUSTOMERS", "60")))
    p.add_argument("--k", type=int, default=int(_env("MEMORYGATE_K", "3")))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="memorygate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("adapters", help="list built-in adapters")

    g = sub.add_parser("generate", help="write the synthetic dataset as JSON")
    _add_dataset_args(g)
    g.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("run", help="score an adapter and print/save the result")
    r.add_argument("--adapter", required=True)
    _add_dataset_args(r)
    r.add_argument("--out", type=Path)

    b = sub.add_parser("baseline", help="score an adapter and save it as the baseline")
    b.add_argument("--adapter", default="better")
    _add_dataset_args(b)
    b.add_argument("--out", type=Path, default=Path(_env("MEMORYGATE_BASELINE", "baselines/baseline.json")))

    t = sub.add_parser("gate", help="compare an adapter to the baseline; non-zero exit on regression")
    t.add_argument("--adapter", required=True)
    t.add_argument("--baseline", type=Path, default=Path(_env("MEMORYGATE_BASELINE", "baselines/baseline.json")))
    t.add_argument("--confidence", type=float, default=0.95)
    t.add_argument("--min-effect", type=float, default=0.01)
    t.add_argument("--bootstrap", type=int, default=2000)
    t.add_argument("--bootstrap-seed", type=int, default=0)
    t.add_argument("--report-md", type=Path)
    t.add_argument("--report-json", type=Path)
    return p


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _score(adapter_spec: str, config: GeneratorConfig, k: int) -> RunResult:
    return run_adapter(load_adapter(adapter_spec), generate(config), k)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (AdapterError, GateError, FileNotFoundError, ValueError) as e:
        print(f"memorygate: error: {e}", file=sys.stderr)
        return EXIT_ERROR


def _dispatch(args: argparse.Namespace) -> int:
    if args.cmd == "adapters":
        print("\n".join(BUILTIN))
        return EXIT_OK
    if args.cmd == "generate":
        ds = generate(GeneratorConfig(seed=args.seed, n_customers=args.customers))
        _write(args.out, ds.model_dump_json(indent=1))
        print(f"wrote {len(ds.conversations)} conversations / {len(ds.queries)} queries "
              f"(fingerprint {ds.fingerprint()}) to {args.out}")
        return EXIT_OK
    if args.cmd in ("run", "baseline"):
        run = _score(args.adapter, GeneratorConfig(seed=args.seed, n_customers=args.customers), args.k)
        print(render_run(run))
        if args.cmd == "baseline":
            _write(args.out, run.model_dump_json(indent=1))
            print(f"baseline saved to {args.out}")
        elif args.out:
            _write(args.out, run.model_dump_json(indent=1))
        return EXIT_OK
    # gate: regenerate exactly the dataset the baseline was recorded on
    if not args.baseline.exists():
        raise FileNotFoundError(f"no baseline at {args.baseline}; run `memorygate baseline` first")
    base = RunResult.model_validate_json(args.baseline.read_text())
    cand = _score(args.adapter, base.dataset_config, base.k)
    report = compare(base, cand, confidence=args.confidence, min_effect=args.min_effect,
                     n_bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    md = render_gate(report)
    print(md)
    if args.report_md:
        _write(args.report_md, md)
    if args.report_json:
        _write(args.report_json, report.model_dump_json(indent=1))
    return EXIT_OK if report.passed else EXIT_REGRESSION


if __name__ == "__main__":
    sys.exit(main())
