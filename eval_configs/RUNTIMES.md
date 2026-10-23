# Runtimes

How long one game takes with a light model on one GPU, to plan allocations.

**One game** is the mean wall-clock time of the N games one command plays one after another (`limit = N`, `max_concurrent = 1`), server start not included. Each game's time comes from the trace timing in `traces.jsonl`: the latest `timing.agent.end` minus the earliest `timing.agent.start` over the game's seat traces. **One game ≈ calls per game × seconds per call**, where calls per game counts the `call` entries (think, talk, answer) in the game's record (`info.game.record`) and seconds per call is one game divided by its calls. Only measured numbers go in the tables. Every game in them was played from start to end by one command; no resumed game is timed.

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
| all: Qwen2.5-7B-Instruct | A40 | 1 min 59 s (N = 1, seed 0) | 30 | 4.0 | coup-qwen7b-13298397, 2026-10-22, with the placeholder answer example `<answer>X</answer>` (1 invalid answer) |
| all: Qwen2.5-7B-Instruct | A40 | 5 min 51 s (N = 5, 1 min 36 s to 16 min 32 s) | 99 (30 to 271) | 3.6 | coup-qwen7b-13320683 (seeds 0 to 3) and coup-qwen7b-d9-s4b (seed 4), 2026-10-22 and 23, same prompts as the row above (1 to 17 invalid answers per game) |

Game settings for every row: the defaults in RULES.md (`talk_rounds` 1, `window_talk_rounds` 0, `thinking` "actor"), temperature 0; seeds 0, 1, 2 in the first two rows, 0 in the third, 0 to 4 in the fourth. In the fourth row seeds 3 and 4 are long games (124 and 271 calls; seeds 0 to 2 took 30 to 36 calls and 1 min 36 s to 2 min 10 s), so its mean says little about a typical game; the model's own share of a call was 3.0 s. For the first two rows, the commands' `real` times were 24 min 30 s and 8 min 49 s for the 3 games, matching the sums of the trace times (24 min 16 s and 8 min 41 s). The model's own share of a call (`calls[].time` in the trace) was 2.8 s and 2.9 s on average; the rest is client and harness time. Invalid answers are decisions (answer and think calls) the runner replaced with a fallback. Fewer of them made the games much shorter (21 to 45 turns before, 5 to 13 after), so the time per call stayed the same and the time per game fell.

## sheriff, 3 players

| Seats | GPU | One game (mean of N, min to max) | Calls/game | Sec/call | Run |
|---|---|---|---|---|---|
| all: Qwen2.5-7B-Instruct | A40 | 16 min 51 s (N = 3, 13 min 28 s to 18 min 42 s) | 316 (285 to 338) | 3.2 | sheriff-qwen7b-13179117, 2026-10-15, with the "Your decision now" prompt line (30.5% invalid) |
| all: Qwen2.5-7B-Instruct | A40 | 15 min 53 s (N = 1, seed 0) | 348 | 2.7 | sheriff-qwen7b-13298397, 2026-10-22, with the placeholder answer example `<answer>X</answer>` (55 invalid answers, all illegal moves) |
| all: Qwen2.5-7B-Instruct | A40 | 25 min 57 s (N = 3, 21 min 38 s to 30 min 6 s) | 526 (476 to 600) | 3.0 | sheriff-qwen7b-d9 (seeds 0, 1) and sheriff-qwen7b-d9-s2 (seed 2), 2026-10-22 and 23, with `negotiation_talk_required = true` and the decision prompts that lead with the player's own state and the offer (17 to 33 invalid answers per game) |

Game settings: the defaults in RULES.md (`talk_rounds` 2, `thinking` "round"), temperature 0; seeds 0, 1, 2 in the first and third rows, 0 in the second. Every game played all 9 rounds. In the third row each game had 27 think, 72 talk and 377 to 501 answer calls, and the model's own share of a call was 2.3 s. The rest of this paragraph is about the first row. The command's `real` time was 50 min 40 s for the 3 games, matching the sum of the trace times (50 min 33 s). The model's own share of a call (`calls[].time`) was 2.5 s on average. Per game: 27 think, 72 talk and 186 to 239 answer calls. Of the 732 decisions (answer and think calls), 223 were replaced with a fallback: 113 illegal actions (often a card no longer in hand), 60 answers without an `<answer>` tag, 18 not readable as JSON, and 32 thinking turns without an `<answer>` tag. The longest prompt was 5,726 tokens (an `offer_goods` answer, seed 0), well inside the 32,768-token context.

## chameleon, 4 players

| Seats | GPU | One game (mean of N, min to max) | Calls/game | Sec/call | Run |
|---|---|---|---|---|---|
| all: Qwen2.5-7B-Instruct | A40 | 1 min 13 s (N = 1, seed 0) | 21 | 3.5 | chameleon-qwen7b-13294724, 2026-10-21 |
| all: Qwen2.5-14B-Instruct | A40 | 2 min 18 s (N = 1, seed 0) | 22 | 6.3 | chameleon-qwen14b-13294724, 2026-10-21 |
| all: Qwen2.5-7B-Instruct | A40 | 1 min 5 s (N = 8, 58 s to 1 min 14 s) | 21.5 (21 to 22) | 3.0 | chameleon-qwen7b-13320683, 2026-10-22, with the clue check and `max_retries` 2 |

Game settings: the defaults in RULES.md (`talk_rounds` 1), temperature 0. The first two rows: seed 0, `max_retries` 0, before the clue check; each is one game, so they say nothing about the spread between games. The third row: seeds 0 to 7; a rejected clue asked again adds one call.

## Server start

Router "Starting router" to "Router ready" in the vLLM log.

| Model | GPU | First start (with download) | Later starts |
|---|---|---|---|
| Qwen2.5-7B-Instruct | A40 | 5 min 1 s, including a 116 s weights download (2026-10-15, job 13170009) | 1 min 30 s, 2 min 1 s and 1 min 30 s (2026-10-15, jobs 13176501, 13177897 and 13179117) |
