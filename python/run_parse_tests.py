#!/usr/bin/env python3
"""
run_parse_tests.py  -  CI test runner for parse_pflare_output.

Runs a subset of small PFLARE advection test cases, captures their output,
calls parse_pflare_output() on it, and asserts:
  - grid complexity < 3.0           (loose bound; won't break on minor changes)
  - reuse_storage complexity == 0.0 (no reuse used in these tests)
  - KSP iterations < ksp_max_it     (solver converged within configured limit)

Designed to run inside the PFLARE Docker image (stevendargaville/pflare:latest)
where the test executables live at /build/PFLARE/tests.
"""

import os
import subprocess
import sys
import tempfile

# parse_pflare_output.py lives in tools/, one level up from python/.
# Override with PFLARE_TOOLS_DIR when the tools directory is not adjacent
# to this script (e.g. when mounted at a different path inside Docker).
_tools_dir = os.environ.get(
    "PFLARE_TOOLS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"),
)
sys.path.insert(0, _tools_dir)
from parse_pflare_output import parse_pflare_output

# Location of test executables.
# Defaults to ../tests/ relative to this script so it works locally.
# Override with PFLARE_TESTS_DIR environment variable when the tests directory
# is not adjacent to this script (e.g. inside the Docker image).
_TESTS_DIR = os.environ.get(
    "PFLARE_TESTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"),
)

# Each entry: (description, command_args_list, ksp_max_it)
# -ksp_monitor is added so that "N KSP Residual norm" lines appear in the
# output; the parser counts iterations from those lines.
TESTS = [
    (
        "adv_diff_fd fc smoothing (8x8)",
        [
            "./adv_diff_fd",
            "-da_grid_x", "8", "-da_grid_y", "8",
            "-pc_type", "air", "-ksp_max_it", "3",
            "-pc_air_smooth_type", "fc",
            "-ksp_monitor",
            "-pc_air_print_stats_timings",
        ],
        3,
    ),
    (
        "adv_diff_fd diag_dom threshold 0.9 (8x8)",
        [
            "./adv_diff_fd",
            "-da_grid_x", "8", "-da_grid_y", "8",
            "-pc_type", "air", "-ksp_max_it", "3",
            "-pc_air_cf_splitting_type", "diag_dom",
            "-pc_air_strong_threshold", "0.9",
            "-ksp_monitor",
            "-pc_air_print_stats_timings",
            "-second_solve",
        ],
        3,
    ),
    (
        "adv_dg_upwind 2D quads",
        [
            "./adv_dg_upwind",
            "-dm_plex_simplex", "0",
            "-pc_type", "air",
            "-ksp_type", "richardson", "-ksp_norm_type", "unpreconditioned",
            "-ksp_max_it", "4",
            "-ksp_monitor",
            "-pc_air_print_stats_timings",
        ],
        4,
    ),
]


def _run_test(desc, cmd, ksp_max_it):
    """Run one test case and return a list of failure strings (empty = pass)."""

    print(f"\n{'=' * 60}", flush=True)
    print(f"Test: {desc}", flush=True)
    print(f"Cmd:  {' '.join(cmd)}", flush=True)
    print("=" * 60, flush=True)

    fd, tmpfile = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    failures = []

    try:
        with open(tmpfile, "w") as out_fh:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=_TESTS_DIR,
                    stdout=out_fh,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                failures.append("test binary timed out after 10 seconds")
                return failures

        if result.returncode != 0:
            # Print captured output so the failure reason is visible in CI logs.
            with open(tmpfile) as fh:
                print(fh.read(), flush=True)
            failures.append(
                f"test binary exited with non-zero code {result.returncode}"
            )
            return failures

        data = parse_pflare_output(tmpfile)

        # ---- Complexity checks ----------------------------------------
        if data["complexities"] is None:
            failures.append("no complexity data found in output")
        else:
            gc = data["complexities"]["grid"]
            rs = data["complexities"]["reuse_storage"]

            if gc >= 3.0:
                failures.append(
                    f"grid complexity {gc:.4f} >= 3.0 (expected < 3.0)"
                )
            else:
                print(f"  OK  grid complexity : {gc:.4f} < 3.0", flush=True)

            if rs != 0.0:
                failures.append(
                    f"reuse_storage complexity {rs} != 0.0 (expected 0.0)"
                )
            else:
                print(f"  OK  reuse_storage   : {rs} == 0.0", flush=True)

        # ---- Iteration count check ------------------------------------
        if not data["ksp_solves"]:
            failures.append(
                "no KSP solve data found — did -ksp_monitor produce output?"
            )
        else:
            for idx, solve in enumerate(data["ksp_solves"]):
                iters = solve["iterations"]
                label = f"solve {idx + 1}"
                if iters is None:
                    failures.append(
                        f"{label}: iteration count is None (no residual norms were parsed)"
                    )
                elif iters >= ksp_max_it:
                    failures.append(
                        f"{label}: iterations {iters} >= ksp_max_it {ksp_max_it} "
                        f"(solver did not converge within limit)"
                    )
                else:
                    print(
                        f"  OK  {label} iterations : {iters} < ksp_max_it={ksp_max_it}",
                        flush=True,
                    )

    finally:
        os.unlink(tmpfile)

    return failures


def main():
    all_failures = {}
    for desc, cmd, ksp_max_it in TESTS:
        fails = _run_test(desc, cmd, ksp_max_it)
        if fails:
            all_failures[desc] = fails

    print(f"\n{'=' * 60}", flush=True)
    if all_failures:
        print(
            f"FAILED: {len(all_failures)}/{len(TESTS)} test(s) had failures:",
            flush=True,
        )
        for desc, fails in all_failures.items():
            print(f"  [{desc}]", flush=True)
            for msg in fails:
                print(f"    - {msg}", flush=True)
        sys.exit(1)
    else:
        print(f"All {len(TESTS)} tests passed.", flush=True)


if __name__ == "__main__":
    main()
