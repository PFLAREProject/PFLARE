"""
parse_pflare_output.py  -  Parse a PFLARE run output file.

Usage:
    python3 parse_pflare_output.py <output_file>

Returns (via parse_pflare_output()):
    {
        'n_levels': int,
        'levels': [
            {
                'level': int,
                'global_rows': int,
                'global_f_points': int or None,   # None for coarse grid
                'global_c_points': int or None,   # None for coarse grid
                'timings': {
                    'coarsen':     float,
                    'extract':     float,
                    'proc agglom': float,
                    'inverse':     float,
                    'restrict':    float,
                    'prolong':     float,
                    'constrain':   float,
                    'rap':         float,
                    'identity':    float,
                    'drop':        float,
                    'truncate':    float,
                }
            },
            ...
        ],
        'total_setup_time': float or None,
        'complexities': {
            'grid':          float,
            'operator':      float,
            'cycle':         float,
            'storage':       float,
            'reuse_storage': float,
        } or None,
        'ksp_solves': [
            {
                'stage':      str or None,
                'time':       float or None,
                'iterations': int or None,
                'residuals':  [float, ...],
            },
            ...
        ] or None,  # None when neither residual norms nor log_view data found
    }

Notes:
  - Timings are stored as per-level values (not cumulative).
  - ksp_solves entries are paired positionally: residual-norm blocks (from
    "N KSP Residual norm" lines) with KSPSolve rows from -log_view output.
  - KSPSolve stage names are recorded as metadata only when present.
"""

import re
import sys

# Timer labels exactly as printed by print_timers(), mapped to dict keys.
_TIMER_LABELS = [
    ('coarsen time',     'coarsen'),
    ('extract time',     'extract'),
    ('proc agglom time', 'proc agglom'),
    ('inverse time',     'inverse'),
    ('restrict time',    'restrict'),
    ('prolong time',     'prolong'),
    ('constrain time',   'constrain'),
    ('rap time',         'rap'),
    ('identity time',    'identity'),
    ('drop time',        'drop'),
    ('truncate time',    'truncate'),
]

_RE_LEVEL    = re.compile(r'~~~~~~~~~~~~ Level\s+(\d+)')
_RE_COARSE   = re.compile(r'~~~~~~~~~~~~ Coarse grid\s+(\d+)')
_RE_GRID_RFC = re.compile(
    r'Global rows\s+(\d+)\s+Global F-points\s+(\d+)\s+Global C-points\s+(\d+)')
_RE_GRID_R   = re.compile(r'Global rows\s+(\d+)')
_RE_TIMER    = {key: re.compile(r'\b' + re.escape(label) + r'\s*:\s*(\S+)')
                for label, key in _TIMER_LABELS}
_RE_TOTAL    = re.compile(r'Total cumulative setup time\s*:\s*(\S+)')
_RE_COMPLEX  = {
    'grid':          re.compile(r'Grid complexity\s*:\s*(\S+)'),
    'operator':      re.compile(r'Operator complexity\s*:\s*(\S+)'),
    'cycle':         re.compile(r'Cycle complexity\s*:\s*(\S+)'),
    'storage':       re.compile(r'Storage complexity\s*:\s*(\S+)'),
    'reuse_storage': re.compile(r'Reuse storage complexity\s*:\s*(\S+)'),
}
_RE_RESIDUAL  = re.compile(r'^\s+(\d+)\s+KSP Residual norm\s+(\S+)')
_RE_STAGE     = re.compile(r'---\s+Event Stage\s+\d+:\s+(.+)')
# KSPSolve row format (from -log_view):
#   KSPSolve  <Count> <CountRatio> <TimeMax> <TimeRatio> ...
_RE_KSPSOLVE  = re.compile(r'^KSPSolve\s+(\d+)\s+\S+\s+(\S+)')


def parse_pflare_output(filename):

    with open(filename, 'r') as fh:
        lines = fh.readlines()

    # ------------------------------------------------------------------
    # Pass 1: collect level blocks with cumulative timer values
    # ------------------------------------------------------------------
    level_blocks = []
    current = None

    for line in lines:
        m = _RE_LEVEL.search(line)
        if m:
            current = {
                'level': int(m.group(1)),
                'global_rows': None,
                'global_f_points': None,
                'global_c_points': None,
                '_cum': {k: None for _, k in _TIMER_LABELS},
            }
            level_blocks.append(current)
            continue

        m = _RE_COARSE.search(line)
        if m:
            current = {
                'level': int(m.group(1)),
                'global_rows': None,
                'global_f_points': None,
                'global_c_points': None,
                '_cum': {k: None for _, k in _TIMER_LABELS},
            }
            level_blocks.append(current)
            continue

        if current is None:
            continue

        # Grid sizes - try the full RFC pattern first, fall back to rows-only
        if current['global_rows'] is None:
            m = _RE_GRID_RFC.search(line)
            if m:
                current['global_rows']     = int(m.group(1))
                current['global_f_points'] = int(m.group(2))
                current['global_c_points'] = int(m.group(3))
            else:
                m = _RE_GRID_R.search(line)
                if m:
                    current['global_rows'] = int(m.group(1))

        # Cumulative timer values
        for label, key in _TIMER_LABELS:
            m = _RE_TIMER[key].search(line)
            if m:
                current['_cum'][key] = float(m.group(1))

    # Convert cumulative timers to per-level incremental deltas
    prev = {k: 0.0 for _, k in _TIMER_LABELS}
    for blk in level_blocks:
        timings = {}
        for _, key in _TIMER_LABELS:
            cum = blk['_cum'][key]
            if cum is None:
                cum = prev[key]   # timer unchanged from previous level; delta = 0
            timings[key] = cum - prev[key]
            prev[key] = cum
        blk['timings'] = timings
        del blk['_cum']

    # ------------------------------------------------------------------
    # Pass 2: total setup time and complexities
    # ------------------------------------------------------------------
    total_setup_time = None
    complexities = {k: None for k in _RE_COMPLEX}

    for line in lines:
        if total_setup_time is None:
            m = _RE_TOTAL.search(line)
            if m:
                total_setup_time = float(m.group(1))

        for key, pat in _RE_COMPLEX.items():
            if complexities[key] is None:
                m = pat.search(line)
                if m:
                    complexities[key] = float(m.group(1))

    all_found = all(v is not None for v in complexities.values())

    # ------------------------------------------------------------------
    # Pass 3: KSP residual norm blocks -> one block per solve
    # A new block starts whenever the iteration counter resets to 0.
    # ------------------------------------------------------------------
    residual_blocks = []
    current_block = None

    for line in lines:
        m = _RE_RESIDUAL.match(line)
        if m:
            it  = int(m.group(1))
            res = float(m.group(2))
            if it == 0 or current_block is None:
                current_block = []
                residual_blocks.append(current_block)
            current_block.append(res)

    # ------------------------------------------------------------------
    # Pass 4: KSPSolve rows from -log_view
    # KSPSolve rows are matched directly; no dependency on event stage
    # headers.  Stage names are recorded as metadata when present.
    # ------------------------------------------------------------------
    ksp_solve_rows = []   # list of {'stage': str|None, 'count': int, 'time': float}
    current_stage = None

    for line in lines:
        m = _RE_STAGE.match(line)
        if m:
            current_stage = m.group(1).strip()
            continue

        m = _RE_KSPSOLVE.match(line)
        if m:
            try:
                count    = int(m.group(1))
                time_sec = float(m.group(2))
            except ValueError:
                continue
            ksp_solve_rows.append({
                'stage': current_stage,
                'count': count,
                'time':  time_sec,
            })

    # ------------------------------------------------------------------
    # Assemble ksp_solves: pair residual blocks with KSPSolve rows
    # positionally (solve 0 <-> index 0, solve 1 <-> index 1, ...).
    # ------------------------------------------------------------------
    ksp_solves = None
    if residual_blocks or ksp_solve_rows:
        ksp_solves = []
        n = max(len(residual_blocks), len(ksp_solve_rows))
        for i in range(n):
            residuals = residual_blocks[i] if i < len(residual_blocks) else []
            if i < len(ksp_solve_rows):
                row      = ksp_solve_rows[i]
                stage    = row['stage']
                time_sec = row['time']
            else:
                stage, time_sec = None, None
            ksp_solves.append({
                'stage':      stage,
                'time':       time_sec,
                'iterations': len(residuals) - 1 if residuals else None,
                'residuals':  residuals,
            })

    return {
        'n_levels':         len(level_blocks),
        'levels':           level_blocks,
        'total_setup_time': total_setup_time,
        'complexities':     complexities if all_found else None,
        'ksp_solves':       ksp_solves,
    }


# ----------------------------------------------------------------------
# Pretty-print helper
# ----------------------------------------------------------------------

def _print_results(data):
    timer_keys = [k for _, k in _TIMER_LABELS]
    col_w = 12

    header = (f"{'Level':>6}  {'Rows':>12}  {'F-pts':>12}  {'C-pts':>12}"
              + ''.join(f'  {k:>{col_w}}' for k in timer_keys))
    print(header)
    print('-' * len(header))

    for lvl in data['levels']:
        f_str = str(lvl['global_f_points']) if lvl['global_f_points'] is not None else '-'
        c_str = str(lvl['global_c_points']) if lvl['global_c_points'] is not None else '-'
        row = (f"{lvl['level']:>6}  {str(lvl['global_rows']):>12}  "
               f"{f_str:>12}  {c_str:>12}"
               + ''.join(f"  {lvl['timings'][k]:>{col_w}.4f}" for k in timer_keys))
        print(row)

    print()
    if data['total_setup_time'] is not None:
        print(f"Total cumulative setup time : {data['total_setup_time']:.6f}")

    print()
    if data['complexities']:
        print("Complexities:")
        for key, val in data['complexities'].items():
            print(f"  {key:<20}: {val:.6f}")

    print()
    if data['ksp_solves']:
        print("KSP solves:")
        for i, s in enumerate(data['ksp_solves']):
            stage  = s['stage'] if s['stage'] else 'unknown'
            t_str  = f"{s['time']:.4f}" if s['time'] is not None else 'n/a'
            it_str = str(s['iterations']) if s['iterations'] is not None else 'n/a'
            print(f"  Solve {i + 1}  stage='{stage}'  time={t_str}s  "
                  f"iterations={it_str}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output_file>", file=sys.stderr)
        sys.exit(1)

    results = parse_pflare_output(sys.argv[1])
    _print_results(results)
