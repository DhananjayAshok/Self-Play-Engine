#!/bin/bash
# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
#SBATCH --job-name=llama-rl
#SBATCH --account=jonmay_231
#SBATCH --partition=isi
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=/project2/jonmay_1426/ashokd/Self-Play/slurm_logs/%x-%A_%a.out

# One line of a sweep file per array task, on 2 A40s (1 vLLM + 1 trainer while training, then
# one eval server per GPU):
#   sbatch --array=1-8 scripts/llama_sweep.sbatch sweeps/<file>.txt
# Line format (blank lines and # comments skipped):
#   <run name> <rl config> [extra rl flags...]   train, then held-out evals at temp 1 and 0
#   base                                         the base model's held-out evals (600 seeds)
module purge
cd /home1/ashokd/projects/Self-Play-Engine || exit 1
source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source prime-rl/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }
export HF_HOME=/project2/jonmay_1426/huggingface_cache
export TMPDIR=/scratch1/$USER/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

line=$(grep -v -E '^\s*(#|$)' "$1" | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r name config overrides <<< "$line"
if [[ -z "$name" ]]; then echo "no line $SLURM_ARRAY_TASK_ID in $1"; exit 1; fi
IFS=',' read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
# A block of 200 ports per job, so jobs sharing a node never collide. prime-rl's vLLM router
# also binds a metrics port at port + 21000, so the base stays below 11768: that keeps
# port + 21000 below Linux's ephemeral range (32768+), where other processes take ports at
# random (a collision there crashed a run), and below 65536 (larger crashed the router).
# Ports used: port (vLLM), +50 (rollout transport), +60/+70 (eval servers), +100 (router
# engine), +21000 (router metrics); the first slot whose ports are all free is taken.
function ports_free() {
    python - "$1" <<'EOF'
import socket, sys
base = int(sys.argv[1])
for p in (base, base + 50, base + 60, base + 70, base + 100, base + 21000):
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", p))
        except OSError:
            sys.exit(1)
EOF
}
for try in $(seq 0 47); do
    port=$((2000 + ((SLURM_JOB_ID + try) % 48) * 200))
    if ports_free "$port"; then break; fi
done
echo "task $SLURM_ARRAY_TASK_ID on $(hostname): name=$name config=$config overrides=[$overrides] gpus=${gpus[*]} port=$port"

if [[ "$name" == "base" ]]; then
    bash scripts/eval_adapters.sh --include_base true --gpu "${gpus[0]}" --port "$((port + 60))" \
        --configs eval_configs/llama3_copycat4_t1.toml,eval_configs/llama3_copycat4_t1_more.toml --tags t1,t1m &
    bash scripts/eval_adapters.sh --include_base true --gpu "${gpus[1]}" --port "$((port + 70))" \
        --configs eval_configs/llama3_copycat4_t0.toml,eval_configs/llama3_copycat4_t0_more.toml --tags t0,t0m &
    wait
else
    overrides_arg="none"
    if [[ -n "$overrides" ]]; then overrides_arg="$overrides"; fi
    bash scripts/rl_run.sh --name "$name" --config "$config" --overrides "$overrides_arg" --port "$port" --save_every 4
    # Training is over either way (finished, collapsed or failed): evaluate what was saved.
    bash scripts/eval_adapters.sh --run "$name" --gpu "${gpus[0]}" --port "$((port + 60))" \
        --configs eval_configs/llama3_copycat4_t1.toml --tags t1 &
    bash scripts/eval_adapters.sh --run "$name" --gpu "${gpus[1]}" --port "$((port + 70))" \
        --configs eval_configs/llama3_copycat4_t0.toml --tags t0 &
    wait
    # Again, one after the other: finished evals are skipped, so this only (re)writes the
    # comparisons, and finishes any eval that was cut short.
    bash scripts/eval_adapters.sh --run "$name" --gpu "${gpus[0]}" --port "$((port + 60))" \
        --configs eval_configs/llama3_copycat4_t1.toml,eval_configs/llama3_copycat4_t0.toml --tags t1,t0
fi
rm -rf "$TMPDIR"
