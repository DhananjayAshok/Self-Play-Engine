# Runtimes

How long one game takes with a light model on one GPU, to plan allocations.

**One game** is the mean wall-clock time of the N games one command plays one after another (`limit = N`, `max_concurrent = 1`), server start not included. Each game's time comes from the trace timing in `traces.jsonl`: the latest `timing.agent.end` minus the earliest `timing.agent.start` over the game's seat traces. **One game ≈ calls per game × seconds per call**, where calls per game counts the `call` entries (think, talk, answer) in the game's record (`info.game.record`) and seconds per call is one game divided by its calls. Only measured numbers go in the tables.

## Run

From the repo root, in one GPU allocation (an A40 fits a 7B model in bf16). GPU nodes are shared, so pick a router port that is free on the node; the engine uses that port + 100.

```bash
salloc --account=<account> --partition=gpu --gres=gpu:a40:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --no-shell
srun --jobid=<id> --overlap bash -c 'ss -ltn | grep -E ":(<port>|<port+100>)\b" || echo free'
# vLLM behind prime-rl's router; the model downloads to $storage_dir/hf
srun --jobid=<id> --overlap bash -c 'source prime-rl/.venv/bin/activate; source configs/config.env;
  export HF_HOME=$storage_dir/hf PRL_OUTPUT_DIR=$storage_dir/inference_outputs;
  inference --vllm.model Qwen/Qwen2.5-7B-Instruct --server.host 0.0.0.0 --server.port <port>' > vllm.log 2>&1 &
srun --jobid=<id> --overlap curl -s http://localhost:<port>/v1/models   # repeat until the model is listed
srun --jobid=<id> --overlap bash -c 'source configs/config.env; export HF_HOME=$storage_dir/hf;
  time bash scripts/eval.sh --config eval_configs/<game config>.toml --name <run> --base_url http://$(hostname):<port>/v1'
scancel <id>
```

## coup, 3 players

| Seats | GPU | One game (mean of N, min to max) | Calls/game | Sec/call | Run |
|---|---|---|---|---|---|
| all: Qwen2.5-7B-Instruct | A40 | 8 min 5 s (N = 3, 3 min 49 s to 13 min 24 s) | 139 (69 to 200) | 3.5 | coup-qwen7b-13176501, 2026-10-15, before the "Your decision now" prompt line (52% invalid) |
| all: Qwen2.5-7B-Instruct | A40 | 2 min 54 s (N = 3, 1 min 44 s to 3 min 50 s) | 49 (30 to 70) | 3.6 | coup-qwen7b-13177897, 2026-10-15, after it (12.5% invalid) |

Game settings for both rows: the defaults in RULES.md (`talk_rounds` 1, `window_talk_rounds` 0, `thinking` "actor"), temperature 0, seeds 0, 1, 2. The commands' `real` times were 24 min 30 s and 8 min 49 s for the 3 games, matching the sums of the trace times (24 min 16 s and 8 min 41 s). The model's own share of a call (`calls[].time` in the trace) was 2.8 s and 2.9 s on average; the rest is client and harness time. Invalid answers are decisions (answer and think calls) the runner replaced with a fallback. Fewer of them made the games much shorter (21 to 45 turns before, 5 to 13 after), so the time per call stayed the same and the time per game fell.

## sheriff, 3 players

| Seats | GPU | One game (mean of N, min to max) | Calls/game | Sec/call | Run |
|---|---|---|---|---|---|
| all: Qwen2.5-7B-Instruct | A40 | 16 min 51 s (N = 3, 13 min 28 s to 18 min 42 s) | 316 (285 to 338) | 3.2 | sheriff-qwen7b-13179117, 2026-10-15, with the "Your decision now" prompt line (30.5% invalid) |

Game settings: the defaults in RULES.md (`talk_rounds` 2, `thinking` "round"), temperature 0, seeds 0, 1, 2. Every game played all 9 rounds. The command's `real` time was 50 min 40 s for the 3 games, matching the sum of the trace times (50 min 33 s). The model's own share of a call (`calls[].time`) was 2.5 s on average. Per game: 27 think, 72 talk and 186 to 239 answer calls. Of the 732 decisions (answer and think calls), 223 were replaced with a fallback: 113 illegal actions (often a card no longer in hand), 60 answers without an `<answer>` tag, 18 not readable as JSON, and 32 thinking turns without an `<answer>` tag. The longest prompt was 5,726 tokens (an `offer_goods` answer, seed 0), well inside the 32,768-token context.

## Server start

Router "Starting router" to "Router ready" in the vLLM log.

| Model | GPU | First start (with download) | Later starts |
|---|---|---|---|
| Qwen2.5-7B-Instruct | A40 | 5 min 1 s, including a 116 s weights download (2026-10-15, job 13170009) | 1 min 30 s, 2 min 1 s and 1 min 30 s (2026-10-15, jobs 13176501, 13177897 and 13179117) |
