from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

import verifiers.v1 as vf

from self_play.chat_model import STOP_STRING, InteractionChatModel
from self_play.checkpoint import checkpoint_path
from self_play.game import THINK
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
    forfeit_on_invalid: bool = False
    """Training reward only: a seat with any invalid answer (a fallback move or an unanswered
    thinking turn) scores `invalid_move_reward` instead of its reward. Without it, RL can learn
    never to answer: in copycat RPS the fallback (always the first legal move) draws every round
    after the first, which beats the base model's sampled play. The game's own outcome is
    unchanged; evals leave this off."""
    invalid_move_reward: float = -1.0
    """The training reward of a forfeited seat (see `forfeit_on_invalid`). Below every game
    reward, so an invalid answer always ranks worst within its group."""
    reward_mode: Literal["game", "moves"] = "game"
    """Training reward only, recorded under this name. "game": the game's outcome reward.
    "moves": the mean step reward over the seat's own non-thinking steps (in copycat RPS, the
    mean round result: 1 win, 0.5 draw, 0 loss), a denser signal that credits every round, not
    just who won more. Games without their own step rewards give every step the outcome, so
    there "moves" equals "game". The forfeit still applies on top."""
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
            outcome_reward = result["outcome"]["rewards"][player_id]
            reward = outcome_reward
            if self.config.reward_mode == "moves":
                move_rewards = [
                    step_reward
                    for step, step_reward in zip(runner.game.steps, result["step_rewards"], strict=True)
                    if step["player_id"] == player_id and step["action"] != THINK
                ]
                reward = sum(move_rewards) / len(move_rewards) if move_rewards else 0.0
            if self.config.forfeit_on_invalid and result["invalid"][player_id] > 0:
                reward = self.config.invalid_move_reward
            # The training reward is named after its mode; the game's own outcome (never
            # forfeited) is a metric in every mode, so runs of either mode compare on it.
            trace.record_reward(self.config.reward_mode, reward)
            trace.record_metric("game_outcome", float(outcome_reward))
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
