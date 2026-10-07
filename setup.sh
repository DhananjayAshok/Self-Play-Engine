#!/bin/bash
# Set up a fresh clone: fetch the submodules prime-rl's environment needs, then build it.
#
# Usage: bash setup.sh
# uv builds the environment at prime-rl/.venv; symlink that elsewhere first if wanted.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }

# Only these are needed to build the environment; prime-rl's other submodules are not.
DEPS=(deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config)

git submodule update --init prime-rl
# Some prime-rl submodules use SSH URLs; rewrite to HTTPS so no GitHub SSH key is needed.
git -C prime-rl -c url."https://github.com/".insteadOf=git@github.com: submodule update --init -- "${DEPS[@]}"

# An interrupted checkout can leave a submodule registered but with no files, which
# `submodule update` does not repair. Restore them from the pinned commit.
for dep in "${DEPS[@]}"; do
    if [[ ! -f "prime-rl/$dep/pyproject.toml" ]]; then
        echo "Restoring missing files in prime-rl/$dep"
        git -C "prime-rl/$dep" checkout HEAD -- .
    fi
done

(cd prime-rl && uv sync --all-extras)

echo "Done. Activate with: source prime-rl/.venv/bin/activate"
