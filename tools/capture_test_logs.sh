#!/usr/bin/env bash
# ~~~~~~~~~~~~~~~~~
# PFLARE - capture golden test logs
#
# Runs every test command in the tests/Makefile run_tests_* groups (and
# run_check) with "-ksp_converged_reason -ksp_monitor" appended to
# PETSC_OPTIONS, and keeps the stdout+stderr of each command in its own log.
# tools/compare_test_logs.py turns the logs into iteration counts, converged
# reasons, residual histories and the -pc_air_print_stats_timings stats, and
# compares two captures (or a capture and a baseline in tests/golden/).
#
# The commands are enumerated the same way make tests_search does it (a make -n
# dry run of the groups, with continuation lines joined), so the set of commands
# is exactly the suite, with the same gates (RUN_LOAD_TESTS, SKIP_SINGLE,
# SKIP_KOKKOS, SKIP_FORTRAN, MPIUNI). Commands that SKIP_SINGLE/SKIP_KOKKOS/
# SKIP_FORTRAN turn into echoes are recorded as skipped.
#
# Usage:
#   tools/capture_test_logs.sh [-r ROOT] [-j JOBS] OUTDIR [GROUP ...]
#     -r ROOT  PFLARE tree to capture (default: the tree this script is in).
#              The tree must already be built (make build_tests).
#     -j JOBS  run up to JOBS test commands at once (default 1)
#     OUTDIR   directory to write the capture to (emptied first)
#     GROUP    tests/Makefile groups to run (default: the tests_search groups
#              plus run_check)
# Environment:
#   GOLDEN_PETSC_OPTIONS  options appended to PETSC_OPTIONS
#                         (default "-ksp_converged_reason -ksp_monitor")
#   GOLDEN_CONFIG         override the configuration name
#   GOLDEN_COMMIT         commit recorded in meta.txt (default: git rev-parse HEAD in ROOT)
#   MAKE                  make to use (default make)
#
# Writes OUTDIR/meta.txt (key=value description of the configuration),
# OUTDIR/commands.tsv (id, group, command) and OUTDIR/logs/<id>.log and
# OUTDIR/logs/<id>.status (exit code or "skipped", then the wall time).
# Exits non-zero if any command failed.
# ~~~~~~~~~~~~~~~~~

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ~~~~~~~~~~~~~~~~~
# Run a single command - used by xargs when running in parallel
# ~~~~~~~~~~~~~~~~~
run_one() {
   local outdir="$1" id="$2"
   local cmd log start end status
   cmd="$(awk -F '\t' -v id="$id" '$1 == id { sub(/^[^\t]*\t[^\t]*\t/, ""); print; exit }' "$outdir/commands.tsv")"
   log="$outdir/logs/$id.log"
   start=$(date +%s.%N)
   (cd "$GOLDEN_ROOT/tests" && /bin/sh -c "$cmd") < /dev/null > "$log" 2>&1
   status=$?
   end=$(date +%s.%N)
   # Commands that SKIP_SINGLE/SKIP_KOKKOS/SKIP_FORTRAN have turned into an echo
   case "$cmd" in
      echo\ *) status=skipped ;;
   esac
   printf '%s %s\n' "$status" "$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", e - s }')" > "$outdir/logs/$id.status"
   if [ "$status" = "0" ] || [ "$status" = "skipped" ]; then
      printf '%-8s %s\n' "$status" "$id"
   else
      printf 'FAIL(%s) %s: %s\n' "$status" "$id" "$cmd"
   fi
}

if [ "${1:-}" = "--run-one" ]; then
   run_one "$2" "$3"
   exit 0
fi

usage() {
   sed -n '/^# Usage:/,/^# Exits/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

root="$(cd "$script_dir/.." && pwd)"
jobs=1
while getopts "r:j:h" opt; do
   case "$opt" in
      r) root="$(cd "$OPTARG" && pwd)" || exit 2 ;;
      j) jobs="$OPTARG" ;;
      h) usage; exit 0 ;;
      *) usage; exit 2 ;;
   esac
done
shift $((OPTIND - 1))
if [ $# -lt 1 ]; then
   usage
   exit 2
fi
outdir="$1"
shift
groups="$*"

make_cmd="${MAKE:-make}"
helper="$script_dir/golden_logs.mk"
make_args=(--no-print-directory -C "$root" -f Makefile -f "$helper")
if [ -n "$groups" ]; then
   make_args+=(GOLDEN_GROUPS="$groups")
fi
if [ -n "${GOLDEN_CONFIG:-}" ]; then
   make_args+=(GOLDEN_CONFIG="$GOLDEN_CONFIG")
fi

rm -rf "$outdir"
mkdir -p "$outdir/logs" || exit 2
outdir="$(cd "$outdir" && pwd)"

# ~~~~~~~~~~~~~~~~~
# Describe the configuration
# ~~~~~~~~~~~~~~~~~
if ! info="$($make_cmd "${make_args[@]}" golden_info)"; then
   echo "ERROR: could not read the configuration of $root" >&2
   exit 2
fi
golden_options="${GOLDEN_PETSC_OPTIONS:--ksp_converged_reason -ksp_monitor}"
export PETSC_OPTIONS="${PETSC_OPTIONS:+$PETSC_OPTIONS }$golden_options"
commit="${GOLDEN_COMMIT:-$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)}"
{
   printf '%s\n' "$info"
   echo "commit=$commit"
   echo "petsc_options=$PETSC_OPTIONS"
   echo "pflare_kokkos_debug=${PFLARE_KOKKOS_DEBUG:-}"
   echo "omp_num_threads=${OMP_NUM_THREADS:-}"
   echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$outdir/meta.txt"

# ~~~~~~~~~~~~~~~~~
# Enumerate the commands, as tests_search does
# ~~~~~~~~~~~~~~~~~
if ! dry_run="$($make_cmd "${make_args[@]}" golden_dry_run)"; then
   echo "ERROR: the dry run of the test groups failed" >&2
   exit 2
fi
printf '%s\n' "$dry_run" | \
   sed -e ':join' -e '/\\$/N' -e 's/\\\n[[:space:]]*/ /' -e 't join' -e 's/^[[:space:]]*[@+-][[:space:]]*//' | \
   awk '
      /^#GOLDEN_GROUP / { group = $2; n = 0; next }
      /(^\.\/)|([[:space:]]\.\/)/ { n++; printf "%s.%03d\t%s\t%s\n", group, n, group, $0 }
   ' > "$outdir/commands.tsv"
n_commands=$(wc -l < "$outdir/commands.tsv" | tr -d '[:space:]')
echo "n_commands=$n_commands" >> "$outdir/meta.txt"
if [ "$n_commands" = "0" ]; then
   echo "ERROR: no test commands found" >&2
   exit 2
fi

echo "Capturing $n_commands test commands from $root into $outdir"
echo "Configuration: $(sed -n 's/^config=//p' "$outdir/meta.txt"), PETSC_OPTIONS=$PETSC_OPTIONS"

# ~~~~~~~~~~~~~~~~~
# Run them
# ~~~~~~~~~~~~~~~~~
export GOLDEN_ROOT="$root"
if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
   export DYLD_FALLBACK_LIBRARY_PATH="$root/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
else
   export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cut -f1 "$outdir/commands.tsv" | \
   xargs -P "$jobs" -n 1 "$script_dir/capture_test_logs.sh" --run-one "$outdir"

# ~~~~~~~~~~~~~~~~~
# Summary
# ~~~~~~~~~~~~~~~~~
n_failed=0
n_skipped=0
failed_ids=""
while IFS= read -r id; do
   status_file="$outdir/logs/$id.status"
   if [ ! -f "$status_file" ]; then
      status=missing
   else
      status="$(cut -d ' ' -f1 "$status_file")"
   fi
   if [ "$status" = "skipped" ]; then
      n_skipped=$((n_skipped + 1))
   elif [ "$status" != "0" ]; then
      n_failed=$((n_failed + 1))
      failed_ids="$failed_ids $id"
   fi
done < <(cut -f1 "$outdir/commands.tsv")

echo "Captured $n_commands test commands ($n_skipped skipped, $n_failed failed) into $outdir"
if [ "$n_failed" != "0" ]; then
   echo "Failed commands:"
   for id in $failed_ids; do
      awk -F '\t' -v id="$id" '$1 == id { print "  " $1 ": " $3 }' "$outdir/commands.tsv"
   done
   exit 1
fi
exit 0
