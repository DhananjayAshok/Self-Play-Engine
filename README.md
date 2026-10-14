# Self-Play-Engine

Multi-agent LLM self-play in strategic social games: train with prime-rl, evaluate with the same code. This README covers setup, then [how to train and evaluate](#training-and-evaluation).

## Two environments

| Environment | Where | What runs in it |
|---|---|---|
| **Project environment** | `prime-rl/.venv` (built by prime-rl) | All project code, tests and scripts: `self_play`, `self_play_verifiers`, `project_utils`, verifiers, prime-rl, `scripts/eval.sh`. |
| **llm-utils environment** | `llm-utils/setup/.venv` | Only [llm-utils](https://github.com/DhananjayAshok/llm-utils/), the batched offline inference toolkit, run through `scripts/llm-utils.sh`. |

The project environment is the one you work in, and the only one you activate yourself. `scripts/llm-utils.sh` switches into the llm-utils environment for one command and back again.

The project code is not a separate install. The root `pyproject.toml` is installed into the project environment as an editable dependency of the `prime-rl` submodule, so editing `self_play/` takes effect immediately.

## Requirements

- **Linux.** The project environment does not work on Windows: verifiers imports `fcntl`, which only exists on Unix. Run the setup on the machine where the code will run.
- [uv](https://docs.astral.sh/uv/getting-started/installation/). It fetches Python 3.12 itself.
- git, and network access to GitHub.

## Setup

All commands run from the project root.

### 1. Clone

```bash
git clone https://github.com/DhananjayAshok/Self-Play-Engine.git
cd Self-Play-Engine
```

### 2. Build the project environment

```bash
bash setup.sh
```

This fetches the `prime-rl` submodule and the four dependencies prime-rl needs (`verifiers`, `renderers`, `prime-envs`, `pydantic-config`), then runs `uv sync --all-extras` inside `prime-rl/`. It rewrites GitHub SSH URLs to HTTPS for that one command, so no SSH key is needed and your git config is not changed. It takes a while the first time.

Check it:

```bash
source prime-rl/.venv/bin/activate
python -c "import self_play, self_play_verifiers, project_utils; print('project environment ok')"
```

If your home directory has a small quota, make the environment live elsewhere before the build: create the target folder on larger storage and symlink `prime-rl/.venv` to it, then run `bash setup.sh`.

### 3. Build the llm-utils environment

Only needed if you will run llm-utils (batched local inference). Skip it otherwise.

```bash
git submodule update --init llm-utils
cd llm-utils/setup
uv sync
cd ../..
```

The environment lands in `llm-utils/setup/.venv`, which is where `scripts/llm-utils.sh` looks for it. To keep it on other storage, symlink `llm-utils/setup/.venv` there before `uv sync`.

### 4. Fill in the config

Edit `configs/private_vars.yaml` and replace every `PLACEHOLDER`. `storage_dir` is mandatory: it must be an absolute path, and runs, models and data are written under it. The HuggingFace entries are only used by `sync_data.py`. See [configs/README.md](configs/README.md) for the other variables.

Then, with the project environment active, generate the shell config that the scripts read:

```bash
source prime-rl/.venv/bin/activate
python configs/create_env_file.py
```

Rerun it whenever a YAML file in `configs/` changes.

### 5. Check everything works

With the project environment active:

```bash
pytest tests/
```

## Using the environments

- **Day to day:** `source prime-rl/.venv/bin/activate` at the start of every shell, from the project root. The scripts in `scripts/` activate it themselves.
- **Never run `uv sync` or `uv run` from the project root.** That would create a third environment there. Only run them inside `prime-rl/` or `llm-utils/setup/`.
- **After changing the package list** in the root `pyproject.toml` (adding or renaming a package), update the project environment: `cd prime-rl && uv sync --all-extras && cd ..`.
- **After changing dependencies** in the root `pyproject.toml`, relock inside prime-rl before syncing: `cd prime-rl && uv lock && uv sync --all-extras && cd ..`.

## Training and evaluation

> CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED (this section, through "On SLURM")

The examples use `meta-llama/Meta-Llama-3-8B-Instruct` on 4-round copycat rock paper scissors. Run everything from the project root; the scripts activate the project environment themselves. Runs are written under `$storage_dir`: training to `rl_runs/<name>/`, evaluation to `eval_runs/<name>/`.

### Training

A training run is LoRA RL with prime-rl, configured by a TOML file in `rl_configs/`. It needs 2 GPUs: one serves the policy with vLLM, the other trains.

```bash
bash scripts/rl_run.sh --name llama-001 --config rl_configs/llama3/copycat_llama3.toml
```

- **Extra prime-rl flags** go in `--overrides`, as one quoted string: `--overrides "--trainer.optim.lr 1e-4 --max-steps 20"`.
- **Adapters:** every 2nd step's LoRA adapter is kept in `rl_runs/<name>/adapters/step_N/` (`--save_every` changes that).
- **Logs:** the launcher log is `rl_runs/<name>.log`; trainer, orchestrator, vLLM and env logs are in `rl_runs/<name>/logs/`.
- **Several runs on one node:** give each its own `--port`, 200 apart, with every base port below 11700. A run uses `port` (vLLM), `port + 50` (rollout transport) and `port + 100` (router engine), and prime-rl's router also binds a metrics port at `port + 21000`. That one must stay below 32768: above 65535 the router crashes with an overflow, and in Linux's ephemeral range (32768+) other processes can already hold it ("address already in use"). `scripts/llama_sweep.sbatch` picks free ports this way automatically.

The training reward is set by the `env.*` keys of the config's `[[orchestrator.train.source]]` block:

| Key | Default | Meaning |
|---|---|---|
| `reward_mode` | `"game"` | `"game"`: the game outcome. `"moves"`: the mean result of the seat's own moves (per round in copycat RPS). |
| `forfeit_on_invalid` | `false` | A game with any invalid answer scores `invalid_move_reward` instead. Keep it on: without it RL learns to never answer and take the fallback move. |
| `invalid_move_reward` | `-1.0` | The reward of a forfeited game. |
| `seat0.sampling.max_tokens` | none | Caps each model call in the game. Set it: the run-level `max_completion_tokens` does not reach the game's calls. |

To train another model, change `[model] name` and set `[orchestrator.renderer] name`: the model's own renderer if prime-rl has one, else `"default"` (the model's chat template). prime-rl can only train architectures it implements (`prime-rl/src/prime_rl/trainer/models/`).

### Evaluation

An evaluation plays held-out games (seeds 100000 and up; training uses 0, 1, 2, ...) against a vLLM server, configured by a TOML file in `eval_configs/`. `llama3_copycat4_t1.toml` and `llama3_copycat4_t0.toml` are 200 games at temperature 1 and 0; the `_more` versions are 400 more seeds to pool with them.

Evaluate the base model and a training run's adapters on one GPU, then compare each adapter with the base on the same seeds:

```bash
bash scripts/eval_adapters.sh --run llama-001 --include_base true \
    --configs eval_configs/llama3_copycat4_t1.toml,eval_configs/llama3_copycat4_t0.toml --tags t1,t0
```

This serves the base model with every saved adapter, runs the evals (named `llama-<tag>-base` and `llama-<tag>-<run>-step<N>`; finished ones are skipped) and writes the comparisons to `rl_runs/<run>/eval_summary_<tag>.txt`. Pick adapters with `--steps 4,8` and the GPU and port with `--gpu` and `--port`.

To evaluate against a server you started yourself, use `scripts/eval.sh` directly:

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000
bash scripts/eval.sh --config eval_configs/llama3_copycat4_t1.toml --name llama-t1-base
bash scripts/eval.sh --config eval_configs/llama3_copycat4_t1.toml --name llama-t1-lora --model <served adapter name>
```

Compare two evals of the same seeds (repeat `--base` and `--treat` to pool runs over different seeds):

```bash
python scripts/compare_evals.py --base $storage_dir/eval_runs/llama-t1-base --treat $storage_dir/eval_runs/llama-t1-lora
```

It reports the mean game reward, wins/draws/losses, per-round win rates and a paired bootstrap 95% CI for the difference.

### On SLURM

`scripts/llama_sweep.sbatch` runs one line of a sweep file per array task on 2 A40s: train, then evaluate the saved adapters at temperature 1 and 0. A line `base` runs only the base model's evals (600 seeds, both temperatures).

```bash
sbatch --array=1-3 scripts/llama_sweep.sbatch sweeps/llama_s0_smoke.txt
```

Sweep file lines are `<run name> <rl config> [extra rl flags...]`. Check the `#SBATCH` account and partition lines before submitting. Job logs go to `$storage_dir/slurm_logs/`.

## API keys

Keys are read from environment variables, never from files in the repo:

- `OPENROUTER_API_KEY` for OpenRouter models.
- `VLLM_API_KEY` for a vLLM server started with `--api-key`. If it is unset, requests are sent with the key `EMPTY`, which a server without `--api-key` accepts.
