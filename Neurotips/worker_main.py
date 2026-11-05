"""Headless compute worker entry (runs under SimNIBS Python).

Mirrors v1: dispatches prepare/optimize/simulate stages via runpy, after
configuring the SimNIBS + CUDA environment.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import configure_simnibs_environment, detect_compute_runtime

BACKENDS = {
    "prepare": "engine.prepare_headandLF",
    "optimize": "engine.optimization_tTIS",
    "simulate": "engine.performones_tTIS",
}


def _print_env():
    simnibs = configure_simnibs_environment()
    compute = detect_compute_runtime()
    print(f"SimNIBS: {simnibs.version} ({simnibs.root})")
    print(f"GPU: {compute.gpu_available} ({compute.device_count} device)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ti-stim-worker")
    parser.add_argument("stage", choices=["env", *sorted(BACKENDS)])
    args, backend_args = parser.parse_known_args(argv)
    if args.stage == "env":
        _print_env(); return 0
    configure_simnibs_environment()
    detect_compute_runtime()
    module = BACKENDS[args.stage]
    sys.argv = [module, *backend_args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
