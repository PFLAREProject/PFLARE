#!/usr/bin/env python3
"""
compare_test_logs.py  -  Summarise and compare golden test logs.

The logs come from tools/capture_test_logs.sh, which runs every test command
in the tests/Makefile groups with -ksp_converged_reason -ksp_monitor and keeps
one log per command. Each log is reduced to a compact record:

    id          <group>.<n>, the n-th command of the group
    group       the tests/Makefile group
    cmd         the command line
    status      exit code, or "skipped" (SKIP_SINGLE/SKIP_KOKKOS)
    reasons     [[prefix, reason, iterations], ...] from the
                "Linear [prefix] solve converged/did not converge due to" lines
    res         residual histories, one list per solve (a new solve starts
                whenever the monitor iteration count goes back to 0); each
                entry is the norm from "N KSP Residual norm", or the list of
                norms on a "-ksp_monitor_true_residual" style line
    stats       the -pc_air_print_stats_timings block parsed with
                tools/parse_pflare_output.py, when the log has one:
                n_levels, levels [[level, rows, f-points, c-points], ...]
                and the complexities (timings are not kept)

Usage:
    compare_test_logs.py summarize CAPTURE [-o OUT.json]
        Write the records of a capture directory as JSON (a baseline).
    compare_test_logs.py compare BASE NEW [options]
        Compare two captures; BASE and NEW are each a capture directory
        or a JSON file written by summarize (e.g. tests/golden/gnu_debug.json).

Commands that run on several MPI ranks can print the monitors of KSPs on
sub-communicators (e.g. comm_self_then_world, a KSP on PETSC_COMM_SELF per
rank) from every rank, and the order the ranks' lines reach the log changes
from run to run. When such a command (mpiexec -n >1) differs only in the order
of its solves, i.e. the sorted residual norms and the sorted converged reasons
still match, it is reported as REORDERED and is not a failure
(--strict-order makes it one).

Comparison rule (fortran_removal_plan.md section 6):
    - every command of BASE must be in NEW, and every command in NEW must
      have succeeded (or been skipped in both)
    - exit status, converged reasons and iteration counts identical
    - residual norms equal to a relative tolerance (--res-rtol, default 1e-8
      in double precision and 1e-4 in single)
    - stats: n_levels and the per level rows, F-points and C-points identical,
      complexities equal to a relative tolerance (--complexity-rtol, 1e-10)
Exits 1 and prints every difference if anything does not match.
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_pflare_output import parse_pflare_output  # noqa: E402

# "  3 KSP Residual norm 1.234567890123e-05"
_RE_RES = re.compile(r'^\s*(\d+) KSP Residual norm\s+(\S+)')
# "  3 KSP preconditioned resid norm 1.2e-05 true resid norm 1.3e-05 ||r(i)||/||b|| 1.4e-06"
_RE_RES_TRUE = re.compile(
    r'^\s*(\d+) KSP (?:\S+ )?resid norm\s+(\S+)\s+true resid norm\s+(\S+)'
    r'(?:\s+\|\|r\(i\)\|\|/\|\|b\|\|\s+(\S+))?')
# "Linear solve converged due to CONVERGED_RTOL iterations 7"
# "Linear mg_coarse_ solve did not converge due to DIVERGED_ITS iterations 3"
_RE_REASON = re.compile(
    r'Linear (?:(\S+) )?solve (?:converged|did not converge) due to (\S+) iterations (\d+)')

_STATS_COMPLEXITIES = ('grid', 'operator', 'cycle', 'storage', 'reuse_storage')

# A command that runs on more than one MPI rank
_RE_MULTI_RANK = re.compile(r"(?:^|\s)\S*(?:mpiexec|mpirun)\s(?:.*?\s)??-(?:n|np)\s+(\d+)")


def _float(text):
    try:
        return float(text)
    except ValueError:
        # Keep anything that isn't a number (e.g. "-nan") as text
        return text


def _norm_cmd(cmd):
    return ' '.join(cmd.split())


# ----------------------------------------------------------------------
# Reading a capture
# ----------------------------------------------------------------------

def parse_log(path):
    """Extract the reasons, residual histories and stats from one log."""
    reasons = []
    res = []
    block = None
    with open(path, 'r', errors='replace') as fh:
        lines = fh.readlines()
    for line in lines:
        m = _RE_RES.match(line)
        if m:
            it, value = int(m.group(1)), _float(m.group(2))
        else:
            m = _RE_RES_TRUE.match(line)
            if m:
                it = int(m.group(1))
                value = [_float(g) for g in m.groups()[1:] if g is not None]
        if m:
            if it == 0 or block is None:
                block = []
                res.append(block)
            block.append(value)
            continue
        m = _RE_REASON.search(line)
        if m:
            reasons.append([m.group(1) or '', m.group(2), int(m.group(3))])

    record = {'reasons': reasons, 'res': res}

    stats = parse_pflare_output(path)
    if stats['n_levels'] > 0 or stats['complexities'] is not None:
        record['stats'] = {
            'n_levels': stats['n_levels'],
            'levels': [[lvl['level'], lvl['global_rows'], lvl['global_f_points'],
                        lvl['global_c_points']] for lvl in stats['levels']],
            'complexities': stats['complexities'],
        }
    return record


def read_capture(capture_dir):
    """Read a capture directory written by capture_test_logs.sh."""
    meta = {}
    with open(os.path.join(capture_dir, 'meta.txt')) as fh:
        for line in fh:
            key, sep, value = line.rstrip('\n').partition('=')
            if sep:
                meta[key] = value
    tests = []
    with open(os.path.join(capture_dir, 'commands.tsv')) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            test_id, group, cmd = line.split('\t', 2)
            record = {'id': test_id, 'group': group, 'cmd': _norm_cmd(cmd)}
            status_path = os.path.join(capture_dir, 'logs', test_id + '.status')
            log_path = os.path.join(capture_dir, 'logs', test_id + '.log')
            if os.path.exists(status_path):
                with open(status_path) as sfh:
                    status = sfh.read().split()[0]
                record['status'] = int(status) if status.lstrip('-').isdigit() else status
            else:
                record['status'] = 'missing'
            if os.path.exists(log_path):
                record.update(parse_log(log_path))
            else:
                record.update({'reasons': [], 'res': []})
            tests.append(record)
    return {'meta': meta, 'tests': tests}


def load(path):
    """Load a capture directory or a JSON summary."""
    if os.path.isdir(path):
        return read_capture(path)
    with open(path) as fh:
        return json.load(fh)


def write_json(data, out):
    """One record per line keeps the file compact but still diffable."""
    out.write('{"meta": ' + json.dumps(data['meta'], sort_keys=True) + ',\n')
    out.write('"tests": [\n')
    out.write(',\n'.join(json.dumps(t, separators=(',', ':')) for t in data['tests']))
    out.write('\n]}\n')


# ----------------------------------------------------------------------
# Comparing
# ----------------------------------------------------------------------

def _keyed(tests):
    """Key each record by (group, command, occurrence of that command in the group)."""
    keyed = {}
    order = []
    seen = {}
    for t in tests:
        base = (t['group'], _norm_cmd(t['cmd']))
        seen[base] = seen.get(base, 0) + 1
        key = base + (seen[base],)
        keyed[key] = t
        order.append(key)
    return keyed, order


def _rel_diff(a, b):
    if a == b:
        return 0.0
    if isinstance(a, str) or isinstance(b, str):
        return math.inf
    if math.isnan(a) or math.isnan(b):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0


def _close(a, b, rtol, atol):
    if a == b:
        return True
    if isinstance(a, str) or isinstance(b, str):
        return False
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def compare_record(base, new, args):
    """Return a list of human readable differences between two records."""
    diffs = []
    if base['status'] != new['status']:
        diffs.append(f"exit status {base['status']} -> {new['status']}")
    if new['status'] not in (0, 'skipped'):
        if not diffs:
            diffs.append(f"exit status {new['status']} (failed in both)")

    # Converged reasons and their iteration counts
    if base['reasons'] != new['reasons']:
        n = max(len(base['reasons']), len(new['reasons']))
        for i in range(n):
            b = base['reasons'][i] if i < len(base['reasons']) else None
            c = new['reasons'][i] if i < len(new['reasons']) else None
            if b != c:
                fmt = lambda r: 'none' if r is None else f"{r[0] or ''}{r[1]} iterations {r[2]}"
                diffs.append(f"solve {i + 1} converged reason: {fmt(b)} -> {fmt(c)}")

    # Residual histories
    if len(base['res']) != len(new['res']):
        diffs.append(f"number of monitored solves {len(base['res'])} -> {len(new['res'])}")
    for i, (b_blk, n_blk) in enumerate(zip(base['res'], new['res'])):
        if len(b_blk) != len(n_blk):
            diffs.append(f"solve {i + 1} iterations {len(b_blk) - 1} -> {len(n_blk) - 1}")
        for it, (b_val, n_val) in enumerate(zip(b_blk, n_blk)):
            b_vals = b_val if isinstance(b_val, list) else [b_val]
            n_vals = n_val if isinstance(n_val, list) else [n_val]
            if len(b_vals) != len(n_vals):
                diffs.append(f"solve {i + 1} iteration {it}: monitor format changed")
                continue
            for b_x, n_x in zip(b_vals, n_vals):
                if not _close(b_x, n_x, args.res_rtol, args.res_atol):
                    diffs.append(f"solve {i + 1} iteration {it} residual {b_x!r} -> {n_x!r} "
                                 f"(rel diff {_rel_diff(b_x, n_x):.3g} > {args.res_rtol:g})")

    # The stats block
    b_stats = base.get('stats')
    n_stats = new.get('stats')
    if (b_stats is None) != (n_stats is None):
        diffs.append(f"stats block {'present' if b_stats else 'absent'} -> "
                     f"{'present' if n_stats else 'absent'}")
    elif b_stats is not None:
        if b_stats['n_levels'] != n_stats['n_levels']:
            diffs.append(f"stats n_levels {b_stats['n_levels']} -> {n_stats['n_levels']}")
        for b_lvl, n_lvl in zip(b_stats['levels'], n_stats['levels']):
            if b_lvl != n_lvl:
                diffs.append(f"stats level [level, rows, F-points, C-points] {b_lvl} -> {n_lvl}")
        b_cx = b_stats.get('complexities') or {}
        n_cx = n_stats.get('complexities') or {}
        for key in _STATS_COMPLEXITIES:
            b_x, n_x = b_cx.get(key), n_cx.get(key)
            if b_x is None and n_x is None:
                continue
            if b_x is None or n_x is None or not _close(b_x, n_x, args.complexity_rtol, 0.0):
                diffs.append(f"stats {key} complexity {b_x!r} -> {n_x!r}")
    return diffs


def _multi_rank(cmd):
    m = _RE_MULTI_RANK.search(cmd)
    return m is not None and int(m.group(1)) > 1


def _sort_key(x):
    # Sort numbers before anything kept as text
    return (1, 0.0, x) if isinstance(x, str) else (0, x, '')


def same_up_to_order(base, new, args):
    """Whether two records only differ in the order their solves were printed."""
    if base['status'] != new['status'] or base.get('stats') != new.get('stats'):
        return False
    if sorted(base['reasons']) != sorted(new['reasons']):
        return False

    def flat(record):
        values = []
        for block in record['res']:
            for value in block:
                values.extend(value if isinstance(value, list) else [value])
        return sorted(values, key=_sort_key)

    b_vals, n_vals = flat(base), flat(new)
    return len(b_vals) == len(n_vals) and \
        all(_close(b, n, args.res_rtol, args.res_atol) for b, n in zip(b_vals, n_vals))


def compare(base, new, args):
    base_keyed, base_order = _keyed(base['tests'])
    new_keyed, new_order = _keyed(new['tests'])

    n_diff = 0
    n_reordered = 0
    n_missing = 0
    n_compared = 0
    n_skipped = 0
    lines = []
    for key in base_order:
        b = base_keyed[key]
        if key not in new_keyed:
            n_missing += 1
            lines.append(f"MISSING {b['id']}: {b['cmd']}")
            continue
        c = new_keyed[key]
        n_compared += 1
        if b['status'] == 'skipped' and c['status'] == 'skipped':
            n_skipped += 1
        diffs = compare_record(b, c, args)
        if diffs and not args.strict_order and _multi_rank(c['cmd']) \
                and same_up_to_order(b, c, args):
            n_reordered += 1
            lines.append(f"REORDERED {c['id']}: same solves printed in a different order "
                         f"by the MPI ranks: {c['cmd']}")
            continue
        if diffs:
            n_diff += 1
            lines.append(f"DIFF {c['id']}: {c['cmd']}")
            shown = diffs if args.max_lines <= 0 else diffs[:args.max_lines]
            lines.extend('    ' + d for d in shown)
            if len(shown) < len(diffs):
                lines.append(f"    ... {len(diffs) - len(shown)} more")

    n_new = 0
    n_new_failed = 0
    for key in new_order:
        if key in base_keyed:
            continue
        c = new_keyed[key]
        n_new += 1
        if c['status'] not in (0, 'skipped'):
            n_new_failed += 1
            lines.append(f"NEW FAILED {c['id']} (exit status {c['status']}): {c['cmd']}")
        else:
            lines.append(f"NEW {c['id']} (no baseline): {c['cmd']}")

    for line in lines:
        print(line)
    print(f"Compared {n_compared} commands ({n_skipped} skipped in both): "
          f"{n_diff} differ, {n_reordered} reordered, {n_missing} missing from the new capture, "
          f"{n_new} new ({n_new_failed} of them failed)")
    print(f"Tolerances: residual rtol {args.res_rtol:g} atol {args.res_atol:g}, "
          f"complexity rtol {args.complexity_rtol:g}")
    failed = n_diff > 0 or n_new_failed > 0 or (n_missing > 0 and not args.allow_missing)
    print("Golden log comparison: " + ("FAILED" if failed else "OK"))
    return 1 if failed else 0


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Summarise and compare golden test logs captured by '
                    'tools/capture_test_logs.sh.')
    sub = parser.add_subparsers(dest='command', required=True)

    p_sum = sub.add_parser('summarize', help='write the records of a capture as JSON')
    p_sum.add_argument('capture', help='capture directory')
    p_sum.add_argument('-o', '--output', help='output JSON file (default stdout)')

    p_cmp = sub.add_parser('compare', help='compare two captures')
    p_cmp.add_argument('base', help='baseline: capture directory or JSON file')
    p_cmp.add_argument('new', help='capture directory or JSON file to check')
    p_cmp.add_argument('--res-rtol', type=float, default=None,
                       help='relative tolerance on residual norms '
                            '(default 1e-8, or 1e-4 if either capture is single precision)')
    p_cmp.add_argument('--res-atol', type=float, default=0.0,
                       help='absolute tolerance on residual norms (default 0)')
    p_cmp.add_argument('--complexity-rtol', type=float, default=1e-10,
                       help='relative tolerance on the complexities (default 1e-10)')
    p_cmp.add_argument('--allow-missing', action='store_true',
                       help='do not fail when a baseline command is not in the new capture')
    p_cmp.add_argument('--strict-order', action='store_true',
                       help='fail multi-rank commands whose solves only differ in order')
    p_cmp.add_argument('--max-lines', type=int, default=20,
                       help='differences printed per command (0 for all, default 20)')

    args = parser.parse_args()

    if args.command == 'summarize':
        data = read_capture(args.capture)
        if args.output:
            with open(args.output, 'w') as fh:
                write_json(data, fh)
        else:
            write_json(data, sys.stdout)
        return 0

    base = load(args.base)
    new = load(args.new)
    if args.res_rtol is None:
        single = 'single' in (base['meta'].get('precision'), new['meta'].get('precision'))
        args.res_rtol = 1e-4 if single else 1e-8
    for name, data in (('base', base), ('new', new)):
        meta = data['meta']
        print(f"{name}: config {meta.get('config', '?')}, commit {meta.get('commit', '?')}, "
              f"{len(data['tests'])} commands")
    if base['meta'].get('config') != new['meta'].get('config'):
        print(f"WARNING: comparing different configurations "
              f"({base['meta'].get('config')} and {new['meta'].get('config')})")
    return compare(base, new, args)


if __name__ == '__main__':
    sys.exit(main())
