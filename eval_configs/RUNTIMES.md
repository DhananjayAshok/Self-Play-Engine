# Runtimes

How long one game takes with a light model on one GPU, to plan allocations.

**One game** is the mean wall-clock time of N games played one at a time, server start not included. The configs play one game per command (`limit = 1`, `max_concurrent = 1`), so each game is one timed command with its own `--name`; mean, min and max are over the N commands. **One game ≈ calls per game × seconds per call**, where calls per game counts the `call` entries (think, talk, answer) in the game's record in `traces.jsonl`. Only measured numbers go in the tables.

## Run

From the repo root, in one GPU allocation (an A40 fits a 7B model in bf16):

```bash
salloc --account=<account> --partition=gpu --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --no-shell
# vLLM behind prime-rl's router; the model downloads to $storage_dir/hf
srun --jobid=<id> --overlap bash -c 'source prime-rl/.venv/bin/activate; source configs/config.env;
  export HF_HOME=$storage_dir/hf PRL_OUTPUT_DIR=$storage_dir/inference_outputs;
  inference --vllm.model Qwen/Qwen2.5-7B-Instruct --server.host 0.0.0.0' > vllm.log 2>&1 &
srun --jobid=<id> --overlap curl -s http://localhost:8000/v1/models   # repeat until the model is listed
srun --jobid=<id> --overlap bash -c 'source configs/config.env; export HF_HOME=$storage_dir/hf;
  time bash scripts/eval.sh --config eval_configs/<game config>.toml --name <run> --base_url http://$(hostname):8000/v1'
scancel <id>
```

## coup, 2 players

| Model | GPU | One game (mean of N, min to max) | Calls/game | Sec/call | Run |
|---|---|---|---|---|---|

Server start (router "Starting router" to "Router ready" in the vLLM log): Qwen2.5-7B-Instruct on an A40, first start 5 min 1 s including a 116 s weights download (2026-10-15, job 13170009); later starts not measured.
