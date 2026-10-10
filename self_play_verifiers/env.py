from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from pydantic import Field

import verifiers.v1 as vf

from self_play.chat_model import STOP_STRING, InteractionChatModel
from self_play.checkpoint import checkpoint_path
from self_play.games import GAMES
from self_play.runner import Runner

SEATS = tuple(f"seat{i}" for i in range(6))
"""Game players map to seats in order: the game's first player is seat0, and so on. Only
the seats a game uses open an interaction, so only they produce traces."""


def seat_default() -> vf.AgentConfig:
    """Every seat: the run's model unless pinned, the package's StatelessHarness (harness
    None = the taskset's default), the local subprocess runtime (the default is a remote
    Prime sandbox), and STOP_STRING as a stop sequence. Overrides deep-merge onto this."""
    return vf.AgentConfig(
        runtime=vf.SubprocessConfig(),
        sampling=vf.SamplingConfig(stop=[STOP_STRING]),
    )


class SelfPlayEnvConfig(vf.EnvConfig):
    seat0: vf.AgentConfig = seat_default()
    seat1: vf.AgentConfig = seat_default()
    seat2: vf.AgentConfig = seat_default()
    seat3: vf.AgentConfig = seat_default()
    seat4: vf.AgentConfig = seat_default()
    seat5: vf.AgentConfig = seat_default()
    game: str = "copycat_rps"
    """Which game, by its name in `self_play.games.GAMES`."""
    game_config: dict[str, Any] = Field(default_factory=dict)
    """Every game setting except the seed, which each task supplies."""
    max_retries: int = Field(0, ge=0)
    """Retries for an unreadable or illegal answer before the fallback action."""
    checkpoint_dir: str | None = None
    """Mid-game checkpoints (eval only; set by scripts/eval.sh). Needs one rollout per task."""


class SelfPlayEnv(vf.Env[SelfPlayEnvConfig]):
    async def start(self):
        if self.config.game not in GAMES:
            raise ValueError(f"unknown game {self.config.game!r}; known: {sorted(GAMES)}")
        if self.config.checkpoint_dir is not None:
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self._active_checkpoints: set[Path] = set()

    async def setup(self, agents):
        # A seat pinned to its own model or endpoint is a fixed opponent: never train on it.
        for name in SEATS:
            spec = getattr(self.config, name)
            if spec.model is not None or spec.client is not None:
                getattr(agents, name).trainable = False

    async def run(self, task, agents):
        game_config = {**self.config.game_config, "seed": task.data.info["seed"]}
        game = GAMES[self.config.game](game_config=game_config)
        if len(game.player_ids) > len(SEATS):
            raise ValueError(f"{game.name} has {len(game.player_ids)} players; at most {len(SEATS)} seats exist")
        seats = dict(zip(game.player_ids, SEATS))
        path = None
        if self.config.checkpoint_dir is not None:
            path = checkpoint_path(checkpoint_dir=self.config.checkpoint_dir, game=game.name, game_config=game_config)
            if path in self._active_checkpoints:
                raise RuntimeError(f"two rollouts of one game share {path}: run with one rollout per task (-r 1)")
            self._active_checkpoints.add(path)
        try:
            async with AsyncExitStack() as stack:
                interactions = {}
                for player_id, seat in seats.items():
                    seat_task = vf.Task(vf.TaskData(idx=task.data.idx, prompt=None))
                    interactions[player_id] = await stack.enter_async_context(getattr(agents, seat).interaction(seat_task))
                models = {player_id: InteractionChatModel(interaction=i) for player_id, i in interactions.items()}
                runner = Runner(game=game, models=models, max_retries=self.config.max_retries, checkpoint_path=path)
                result = await runner.run()
        finally:
            self._active_checkpoints.discard(path)

        game_record = {
            "game": runner.game.name,
            "game_config": runner.game.game_config,
            "seats": seats,
            "resumed": runner.resumed,
            "outcome": result["outcome"],
            "steps": runner.game.steps,
            "step_rewards": result["step_rewards"],
            "invalid": result["invalid"],
            "record": runner.record,
        }
        call_steps = runner.call_step_ids()
        calls = [entry for entry in runner.record if entry["kind"] == "call"]
        for player_id, interaction in interactions.items():
            trace = interaction.trace
            trace.record_reward("game", result["outcome"]["rewards"][player_id])
            trace.record_metric("invalid", float(result["invalid"][player_id]))
            # Every seat carries the whole game; step_rewards lists this player's calls in
            # call order with the step each led to and that step's reward.
            trace.info["game"] = {**game_record, "player_id": player_id}
            trace.info["step_rewards"] = [
                {
                    "call": entry["call"],
                    "step_id": step_id,
                    "reward": result["step_rewards"][step_id] if step_id is not None else None,
                }
                for entry, step_id in zip(calls, call_steps, strict=True)
                if entry["player_id"] == player_id
            ]
