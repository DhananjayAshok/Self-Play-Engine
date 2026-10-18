# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
# Per run: best and final held-out t1 / t0 mean (with exploit), from the eval summaries that
# scripts/eval_adapters.sh writes to $storage_dir/rl_runs/<run>/eval_summary_{t1,t0}.txt.
#   bash tmp_best.sh <run name> [<run name> ...]
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
r=$storage_dir/rl_runs
printf "%-34s | %-26s | %-14s | %-26s | %-14s\n" run "t1 best (step, exploit)" "t1 final" "t0 best (step, exploit)" "t0 final"
for run in "$@"; do
  row="$run"
  for tag in t1 t0; do
    f=$r/$run/eval_summary_$tag.txt
    if [[ ! -s $f ]]; then row="$row|n/a|n/a"; continue; fi
    stats=$(awk '
      /^=== /{name=$3; sub(".*-step", "", name)}
      /^treat:/{tr=$4; getline; getline; getline; ex=$NF; print name, tr, ex}' $f | sort -n)
    best=$(echo "$stats" | sort -k2,2nr | head -1)
    last=$(echo "$stats" | tail -1)
    row="$row|$(echo $best | awk '{printf "%s (step %s, %s)", $2, $1, $3}')|$(echo $last | awk '{printf "%s (%s)", $2, $3}')"
  done
  echo "$row" | awk -F'|' '{printf "%-34s | %-26s | %-14s | %-26s | %-14s\n", $1, $2, $3, $4, $5}'
done
