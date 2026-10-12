#!/usr/bin/env bash
# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
# Held-out evals of an RL run's saved adapters (and/or the base model) on one GPU: serves the
# base model with the adapters as LoRA modules, runs scripts/eval.sh for every (config,
# target) pair, then writes a paired comparison of each adapter against the base eval.
#
#   bash scripts/eval_adapters.sh --configs eval_configs/llama3_copycat4_t1.toml --tags t1 --include_base true
#   bash scripts/eval_adapters.sh --run llama-001 --configs <cfg1>,<cfg2> --tags t1,t0 --steps 4,8 --port 20160 --gpu 1
#
# Eval runs are named <prefix>-<tag>-base and <prefix>-<tag>-<run>-step<N> under
# $storage_dir/eval_runs/; finished ones are skipped. The comparison of <run> against
# <prefix>-<tag>-base goes to $storage_dir/rl_runs/<run>/eval_summary_<tag>.txt.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=("configs" "tags")
ARGS["run"]="none"            # RL run whose adapters/step_N to evaluate; none = base only
ARGS["steps"]="all"           # comma list of adapter steps, or all
ARGS["include_base"]="false"  # also evaluate the base model
ARGS["model"]="meta-llama/Meta-Llama-3-8B-Instruct"
ARGS["prefix"]="llama"
ARGS["port"]="8000"
ARGS["gpu"]="0"
ARGS["parallel"]="4"          # evals in flight at once against the server

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
run="${ARGS["run"]}"
port="${ARGS["port"]}"
prefix="${ARGS["prefix"]}"
eval_dir="$storage_dir/eval_runs"
IFS=',' read -ra configs <<< "${ARGS["configs"]}"
IFS=',' read -ra tags <<< "${ARGS["tags"]}"
if [[ ${#configs[@]} -ne ${#tags[@]} ]]; then echo "Error: --configs and --tags must have the same length"; exit 1; fi
include_base=false
if [[ "${ARGS["include_base"]}" == "true" || "${ARGS["include_base"]}" == "yes" || "${ARGS["include_base"]}" == "y" ]]; then include_base=true; fi

# Targets: (served model name, eval run suffix).
served=()
suffixes=()
lora_args=()
if [[ "$include_base" == true ]]; then
    served+=("none")
    suffixes+=("base")
fi
if [[ "$run" != "none" ]]; then
    adapters="$storage_dir/rl_runs/$run/adapters"
    if [[ "${ARGS["steps"]}" == "all" ]]; then
        steps=$(ls "$adapters" | sed -n 's/^step_\([0-9]*\)$/\1/p' | sort -n)
    else
        steps=$(tr ',' ' ' <<< "${ARGS["steps"]}")
    fi
    modules=()
    for step in $steps; do
        if [[ ! -f "$adapters/step_$step/adapter_model.safetensors" ]]; then echo "missing adapter $adapters/step_$step"; continue; fi
        modules+=("step$step=$adapters/step_$step")
        served+=("step$step")
        suffixes+=("$run-step$step")
    done
    if [[ ${#modules[@]} -gt 0 ]]; then lora_args=(--enable-lora --max-lora-rank 64 --max-loras 4 --max-cpu-loras 64 --lora-modules "${modules[@]}"); fi
fi
if [[ ${#served[@]} -eq 0 ]]; then echo "Nothing to evaluate"; exit 1; fi

# Skip targets whose evals all finished already.
todo=()
for i in "${!served[@]}"; do
    for tag in "${tags[@]}"; do
        if [[ ! -s "$eval_dir/$prefix-$tag-${suffixes[$i]}/traces.jsonl" ]]; then todo+=("$i"); break; fi
    done
done

mkdir -p "$eval_dir"
server_pid=""
if [[ ${#todo[@]} -gt 0 ]]; then
    server_log="$eval_dir/$prefix-serve-${run}-$port.log"
    # FlashInfer's sampler JIT-compiles at warmup and needs nvcc; the torch sampler is fine.
    CUDA_VISIBLE_DEVICES="${ARGS["gpu"]}" VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "${ARGS["model"]}" \
        --host 127.0.0.1 --port "$port" --max-model-len 4096 --gpu-memory-utilization 0.9 \
        "${lora_args[@]}" > "$server_log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; then break; fi
        if ! kill -0 "$server_pid" 2> /dev/null; then echo "vLLM server died, see $server_log"; exit 1; fi
        sleep 10
    done
    if ! curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; then echo "vLLM server never came up, see $server_log"; kill "$server_pid"; exit 1; fi

    for i in "${todo[@]}"; do
        for j in "${!configs[@]}"; do
            name="$prefix-${tags[$j]}-${suffixes[$i]}"
            if [[ -s "$eval_dir/$name/traces.jsonl" ]]; then continue; fi
            while [[ $(jobs -rp | wc -l) -gt ${ARGS["parallel"]} ]]; do wait -n; done
            model_args=()
            if [[ "${served[$i]}" != "none" ]]; then model_args=(--model "${served[$i]}"); fi
            echo "eval $name"
            bash scripts/eval.sh --config "${configs[$j]}" --name "$name" --base_url "http://127.0.0.1:$port/v1" "${model_args[@]}" > "$eval_dir/$name.log" 2>&1 &
        done
    done
    while [[ $(jobs -rp | wc -l) -gt 1 ]]; do wait -n; done
    kill "$server_pid"
    wait "$server_pid" 2> /dev/null
fi

# Paired comparisons against the base eval of the same config.
if [[ "$run" != "none" ]]; then
    for tag in "${tags[@]}"; do
        base_eval="$eval_dir/$prefix-$tag-base"
        summary="$storage_dir/rl_runs/$run/eval_summary_$tag.txt"
        if [[ ! -s "$base_eval/traces.jsonl" ]]; then echo "no base eval at $base_eval yet; skipping the $tag comparison"; continue; fi
        : > "$summary"
        for suffix in "${suffixes[@]}"; do
            [[ "$suffix" == "base" ]] && continue
            treat="$eval_dir/$prefix-$tag-$suffix"
            [[ -s "$treat/traces.jsonl" ]] || continue
            echo "=== $tag $suffix" >> "$summary"
            python scripts/compare_evals.py --base "$base_eval" --treat "$treat" >> "$summary" 2>&1
        done
        echo "wrote $summary"
    done
fi
