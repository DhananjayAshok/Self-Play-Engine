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
    assert set(trace.rewards) == {"game"} and set(trace.metrics) == {"game_outcome", "invalid"}
    assert trace.metrics["invalid"] == 0.0 and trace.metrics["game_outcome"] == trace.rewards["game"]
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


# CLAUDE GENERATED (every test below), DJ DOES NOT APPROVE, MUST BE REVISED

def run_seat0(*, replies, **extra):
    env = make_env(**extra)
    interaction = FakeInteraction(replies=replies)
    asyncio.run(env.run(TASK, fake_agents(interaction=interaction)))
    return interaction.trace


def test_call_credit_is_off_by_default():
    info = run_seat0(replies=REPLIES).info
    assert not {"call_rewards", "call_units", "call_tags"} & set(info)


def test_segments_give_each_call_its_rounds_result_and_its_kind_and_phase():
    """Each think opens a segment; a segment's calls get the result of the move made in it."""
    trace = run_seat0(replies=REPLIES, call_credit="segments")
    step_rewards = trace.info["game"]["step_rewards"]  # steps: think 0, scissors 1, think 2, rock 3
    assert trace.info["call_units"] == [0, 0, 0, 1, 1]
    assert trace.info["call_rewards"] == [step_rewards[1]] * 3 + [step_rewards[3]] * 2
    assert trace.info["call_tags"] == ["think:thinking", "talk:boast", "answer:boast", "think:thinking", "answer:comment"]


def test_an_invalid_answer_alone_gets_the_invalid_reward():
    """Round 2's answer is unreadable: the runner plays the fallback, the answer gets -1, and
    the think beside it keeps the round's result (the fallback move's reward)."""
    replies = REPLIES[:4] + [reply(answer="banana")]
    trace = run_seat0(replies=replies, call_credit="segments", forfeit_on_invalid=True)
    step_rewards = trace.info["game"]["step_rewards"]
    assert trace.info["call_rewards"] == [step_rewards[1]] * 3 + [step_rewards[3], -1.0]
    assert trace.rewards == {"game": -1.0} and trace.metrics["invalid"] == 1.0  # the scalar forfeit


def test_mean_steps_is_the_mean_reward_of_the_seats_moves():
    trace = run_seat0(replies=REPLIES, reward_mode="mean_steps")
    step_rewards = trace.info["game"]["step_rewards"]
    assert set(trace.rewards) == {"mean_steps"}
    assert trace.rewards["mean_steps"] == (step_rewards[1] + step_rewards[3]) / 2


def test_segments_and_tags_in_a_scripted_coup_game():
    """Coup, 2 players (the scripted game of tests/test_coup.py): p1's first Coup is forced (no
    call), p2 reveals Duke and takes Income, p1 chooses a second Coup, p2's last reveal is forced.
    Every step earns the final outcome, so each segment's mean is the outcome."""
    from self_play.chat_model import ScriptedChatModel
    from self_play.games.coup import CoupGame
    from self_play.runner import Runner

    game = CoupGame(game_config={"seed": 0, "players": 2, "talk_rounds": 0})
    game.state["actor"] = "p1"
    game.state["coins"] = {"p1": 14, "p2": 2}
    game.state["hands"]["p2"] = ["duke", "captain"]
    models = {
        "p1": ScriptedChatModel(replies=[reply(answer="plan"), reply(answer="plan again"),
                                         reply(answer='{"action": "coup", "target": "p2"}')]),
        "p2": ScriptedChatModel(replies=[reply(answer="plan"), reply(answer='{"reveal": "duke"}'),
                                         reply(answer="plan again"), reply(answer='{"action": "income"}')]),
    }
    runner = Runner(game=game, models=models)
    result = asyncio.run(runner.run())
    env = make_env(game="coup", game_config={"players": 2}, call_credit="segments")
    credit = {pid: env._segment_rewards(runner=runner, result=result, player_id=pid) for pid in ("p1", "p2")}
    tags = {pid: SelfPlayEnv._call_tags(runner=runner, player_id=pid) for pid in ("p1", "p2")}
    assert credit["p1"] == ([1.0, 1.0, 1.0], [0, 1, 1])
    assert credit["p2"] == ([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1])
    assert tags["p1"] == ["think:thinking", "think:thinking", "answer:action"]
    assert tags["p2"] == ["think:thinking", "answer:lose_influence", "think:thinking", "answer:action"]
