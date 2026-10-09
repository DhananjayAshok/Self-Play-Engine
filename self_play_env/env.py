from pydantic import Field

import verifiers.v1 as vf

from self_play.chat_model import STOP_STRING
from self_play_env.chat_model import InteractionChatModel
from self_play_env.smoke_game import play_smoke_game

SEATS = ("seat0", "seat1")


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
    rounds: int = Field(3, ge=1)
    """Rounds of the smoke game."""


class SelfPlayEnv(vf.Env[SelfPlayEnvConfig]):
    async def setup(self, agents):
        # A seat pinned to its own model or endpoint is a fixed opponent: never train on it.
        for name in SEATS:
            spec = getattr(self.config, name)
            if spec.model is not None or spec.client is not None:
                getattr(agents, name).trainable = False

    async def run(self, task, agents):
        # Seat tasks carry no prompt or system prompt: each turn's messages are the whole context.
        seat_tasks = [vf.Task(vf.TaskData(idx=task.data.idx, prompt=None)) for _ in SEATS]
        async with (
            agents.seat0.interaction(seat_tasks[0]) as seat0,
            agents.seat1.interaction(seat_tasks[1]) as seat1,
        ):
            models = [InteractionChatModel(interaction=seat0), InteractionChatModel(interaction=seat1)]
            result = await play_smoke_game(models=models, rounds=self.config.rounds)
        scores = result.scores()
        for seat, trace in enumerate([seat0.trace, seat1.trace]):
            record = result.seats[seat]
            trace.record_reward("win", scores[seat])
            trace.record_metric("invalid", float(record.invalid))
            trace.record_metric("terminated", float(models[seat].terminated))
            trace.info["smoke"] = {
                "seat": seat,
                "seed": task.data.info["seed"],
                "choices": record.choices,
                "replies": record.replies,
                "round_winners": result.round_winners,
            }
