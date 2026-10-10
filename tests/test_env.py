"""SelfPlayEnv end to end with fake seats in place of verifiers' agents. Needs verifiers
(Linux, prime-rl's environment); skipped elsewhere."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers.v1")

from self_play.chat_model import ModelCallFailed  # noqa: E402
from self_play_verifiers.env import SelfPlayEnv, SelfPlayEnvConfig  # noqa: E402
from self_play_verifiers.taskset import SelfPlayData, SelfPlayTask  # noqa: E402


def reply(*, answer):
    return f"<thinking>scratch</thinking><answer>{answer}</answer>[STOP]"


# Copycat over 2 rounds: think, talk + move (ACTION), think, move (RESPONSE_WINDOW).
REPLIES = [reply(answer="notes"), reply(answer="SILENCE"), reply(answer="scissors"),
           reply(answer="NO CHANGE"), reply(answer="rock")]


class FakeTrace:
    def __init__(self):
        self.rewards, self.metrics, self.info = {}, {}, {}

    def record_reward(self, name, value, weight=1.0):
        self.rewards[name] = value

    def record_metric(self, name, value):
        self.metrics[name] = value


class FakeInteraction:
    """Replies in order; after `fail_at` replies the seat's run ends (a terminated segment)."""

    def __init__(self, *, replies, fail_at=None):
        self.replies, self.fail_at, self.turns = list(replies), fail_at, 0
        self.trace = FakeTrace()

    async def turn(self, messages):
        if self.turns == self.fail_at:
            return SimpleNamespace(terminated=True, last_reply="")
        self.turns += 1
        return SimpleNamespace(terminated=False, last_reply=self.replies.pop(0))


class FakeAgent:
    def __init__(self, *, interaction=None):
        self.trainable = True
        self.opened = False
        self._interaction = interaction

    @asynccontextmanager
    async def interaction(self, task):
        self.opened = True
        yield self._interaction


def fake_agents(*, interaction):
    agents = SimpleNamespace(**{f"seat{i}": FakeAgent() for i in range(6)})
    agents.seat0 = FakeAgent(interaction=interaction)
    return agents


def make_env(*, checkpoint_dir=None, **extra):
    config = SelfPlayEnvConfig.model_validate({
        "taskset": {"id": "self-play-verifiers"},
        "game": "copycat_rps",
        "game_config": {"rounds": 2},
        "checkpoint_dir": checkpoint_dir,
        **extra,
    })
    env = SelfPlayEnv(config)
    asyncio.run(env.start())
    return env


TASK = SelfPlayTask(SelfPlayData(name="game#0", prompt=None, info={"seed": 0}))


def test_a_game_runs_through_seat0_and_fills_its_trace():
    env = make_env()
    interaction = FakeInteraction(replies=REPLIES)
    agents = fake_agents(interaction=interaction)
    asyncio.run(env.run(TASK, agents))
    trace = interaction.trace
    assert interaction.turns == len(REPLIES)
    assert set(trace.rewards) == {"game"} and trace.metrics == {"invalid": 0.0}
    game = trace.info["game"]
    assert game["game"] == "copycat_rps" and game["game_config"] == {"rounds": 2, "seed": 0}
    assert game["seats"] == {"player": "seat0"} and game["player_id"] == "player"
    assert [e["call"] for e in game["record"] if e["kind"] == "call"] == ["think", "talk", "answer", "think", "answer"]
    assert [(s["call"], s["step_id"]) for s in trace.info["step_rewards"]] == [
        ("think", 0), ("talk", 1), ("answer", 1), ("think", 2), ("answer", 3)
    ]
    assert not any(getattr(agents, f"seat{i}").opened for i in range(1, 6))  # unused seats stay closed


def test_a_crashed_game_resumes_from_its_checkpoint(tmp_path):
    env = make_env(checkpoint_dir=str(tmp_path))
    with pytest.raises(ModelCallFailed):
        asyncio.run(env.run(TASK, fake_agents(interaction=FakeInteraction(replies=REPLIES, fail_at=3))))
    assert len(list(tmp_path.glob("*.json"))) == 1 and len(list(tmp_path.glob("meta_*.yaml"))) == 1

    rest = FakeInteraction(replies=REPLIES[3:])
    asyncio.run(env.run(TASK, fake_agents(interaction=rest)))
    assert rest.turns == len(REPLIES) - 3  # only the calls the crash left unmade
    game = rest.trace.info["game"]
    assert game["resumed"]
    reference = FakeInteraction(replies=REPLIES)
    asyncio.run(make_env().run(TASK, fake_agents(interaction=reference)))
    assert game["record"] == reference.trace.info["game"]["record"]
    assert rest.trace.rewards == reference.trace.rewards


def test_pinned_seats_are_not_trainable():
    env = make_env(seat1={"model": "some/other-model"})
    agents = fake_agents(interaction=None)
    asyncio.run(env.setup(agents))
    assert agents.seat0.trainable and not agents.seat1.trainable
