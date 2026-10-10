"""Checkpoint files: one JSON file per game, named by a hash of everything that defines
the game, written atomically so a crash leaves the old file or the new one, never half.

    <checkpoint_dir>/<hash>.json         the snapshot (see `Runner`)
    <checkpoint_dir>/meta_<hash>.yaml    the game and game_config the hash stands for
"""

import json
import os
from pathlib import Path
from typing import Any

from project_utils.hash_handling import hash_meta_dict, write_meta


def game_key(*, game: str, game_config: dict[str, Any]) -> dict[str, str]:
    """What identifies a game. game_config is sorted JSON so its key order never matters."""
    return {"game": game, "game_config": json.dumps(game_config, sort_keys=True)}


def checkpoint_path(*, checkpoint_dir: str | Path, game: str, game_config: dict[str, Any]) -> Path:
    """The game's snapshot path; writes its meta file the first time."""
    key = game_key(game=game, game_config=game_config)
    name = hash_meta_dict(key)
    if not (Path(checkpoint_dir) / f"meta_{name}.yaml").exists():
        write_meta(str(checkpoint_dir), dict(key))  # write_meta adds a timestamp to what it is given
    return Path(checkpoint_dir) / f"{name}.json"


def save_snapshot(*, path: Path, snapshot: dict[str, Any]) -> None:
    """Write via a temporary file and an atomic rename."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot))
    os.replace(tmp, path)


def load_snapshot(*, path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None
