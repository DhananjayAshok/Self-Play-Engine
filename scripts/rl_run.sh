#!/usr/bin/env bash
# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
# One prime-rl LoRA RL run, keeping every `save_every`-th adapter (broadcasts/ keeps only the
# last 2) in $storage_dir/rl_runs/<name>/adapters/step_N/. Blocks until the run ends.
#
#   bash scripts/rl_run.sh --name llama-001 --config rl_configs/llama3/copycat_llama3.toml
#   bash scripts/rl_run.sh --name llama-002 --config <cfg> --overrides "--trainer.optim.lr 1e-4" --port 8010
#
# `port` is the run's vLLM port and `port + 50` its rollout transport (ZMQ) port: give each
# run sharing a node its own block of 100. `overrides` are extra `rl` flags, one quoted string.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("name" "config")
ARGS["overrides"]="none"
ARGS["save_every"]="2"
ARGS["port"]="8000"

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Script logic below:
name="${ARGS["name"]}"
port="${ARGS["port"]}"
save_every="${ARGS["save_every"]}"
run_dir="$storage_dir/rl_runs/$name"
override_args=()
if [[ "${ARGS["overrides"]}" != "none" ]]; then read -ra override_args <<< "${ARGS["overrides"]}"; fi

# Keep every save_every-th finished adapter as it appears.
function save_adapters() {
    for src in "$run_dir"/broadcasts/step_*; do
        [[ -f "$src/.finished" ]] || continue
        step=${src##*_}
        (( step % save_every == 0 )) || continue
        dst="$run_dir/adapters/step_$step"
        if [[ ! -f "$dst/adapter_model.safetensors" ]]; then
            mkdir -p "$dst"
            cp "$src/adapter_config.json" "$src/adapter_model.safetensors" "$dst/"
            echo "saved adapter step $step"
        fi
    done
}

config_path=$(realpath "${ARGS["config"]}")
cd prime-rl || exit 1
rl @ "$config_path" --output-dir "$storage_dir/rl_runs" --run.name "$name" \
    --inference.server.port "$port" --orchestrator.model.client.base-url "http://localhost:$port/v1" \
    --trainer.rollout-transport.port "$((port + 50))" --orchestrator.rollout-transport.port "$((port + 50))" \
    "${override_args[@]}" > "$storage_dir/rl_runs/$name.log" 2>&1 &
rl_pid=$!
cd "$PROJECT_ROOT" || exit 1

while kill -0 "$rl_pid" 2> /dev/null; do
    save_adapters
    sleep 15
done
wait "$rl_pid"
status=$?
save_adapters
echo "rl exited with status $status"
exit $status
