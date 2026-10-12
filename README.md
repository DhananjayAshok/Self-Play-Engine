# Self-Play-Engine

Multi-agent LLM self-play in strategic social games: train with prime-rl, evaluate with the same code. This README covers setup only.

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

## API keys

Keys are read from environment variables, never from files in the repo:

- `OPENROUTER_API_KEY` for OpenRouter models.
- `VLLM_API_KEY` for a vLLM server started with `--api-key`. If it is unset, requests are sent with the key `EMPTY`, which a server without `--api-key` accepts.
