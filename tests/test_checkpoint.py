import asyncio
import json

import pytest

from self_play.chat_model import ChatModel, ModelCallFailed, ScriptedChatModel
from self_play.checkpoint import checkpoint_path
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.copycat_rps.game import PLAYER
from self_play.runner import Runner


def reply(*, answer):
    return f"<thinking>scratch</thinking><answer>{answer}</answer>[STOP]"


# Four rounds cover every mode (ACTION, RESPONSE_WINDOW, DISCUSSION, SIMULTANEOUS), talk,
# thinking turns that change and keep the notes, and an invalid answer.
REPLIES = [
    reply(answer="first notes"),
    reply(answer="Hello."), reply(answer="rock"),          # round 1: talk, move
    reply(answer="NO CHANGE"), reply(answer="lizard"),     # round 2: invalid move, falls back
    reply(answer="second notes"),
    reply(answer="SILENCE"), reply(answer="scissors"),     # round 3: talk, move
    reply(answer="NO CHANGE"), reply(answer="rock"),       # round 4: simultaneous move
]


class CrashingChatModel(ChatModel):
    """Replies like ScriptedChatModel, then fails on call number `fail_at` (from 0)."""

    def __init__(self, *, replies, fail_at):
        self.inner = ScriptedChatModel(replies=replies)
        self.fail_at = fail_at

    async def complete(self, *, messages):
        if len(self.inner.calls) == self.fail_at:
            raise ModelCallFailed("simulated crash")
        return await self.inner.complete(messages=messages)


def new_game():
    game = CopycatRPSGame(game_config={"seed": 0, "rounds": 4})
    return game


def play(*, model, path=None):
    runner = Runner(game=new_game(), models={PLAYER: model}, checkpoint_path=path)
    result = asyncio.run(runner.run())
    return runner, result


def test_the_reference_game_uses_every_reply():
    model = ScriptedChatModel(replies=REPLIES)
    _, result = play(model=model)
    assert len(model.calls) == len(REPLIES)
    assert result["invalid"] == {PLAYER: 1}


@pytest.mark.parametrize("fail_at", range(len(REPLIES)))
def test_a_crash_at_any_call_resumes_to_the_identical_game(tmp_path, fail_at):
    reference, expected = play(model=ScriptedChatModel(replies=REPLIES))
    path = tmp_path / "game.json"

    with pytest.raises(ModelCallFailed):
        play(model=CrashingChatModel(replies=REPLIES, fail_at=fail_at), path=path)

    remaining = ScriptedChatModel(replies=REPLIES[fail_at:])
    resumed, result = play(model=remaining, path=path)
    assert resumed.resumed
    assert len(remaining.calls) == len(REPLIES) - fail_at  # nothing asked twice, nothing skipped
    assert result == expected
    assert resumed.record == json.loads(json.dumps(reference.record))
    assert resumed.game.to_dict() == reference.game.to_dict()
    assert resumed.notes(player_id=PLAYER) == "second notes"


def test_a_finished_game_is_never_played_again(tmp_path):
    path = tmp_path / "game.json"
    first, expected = play(model=ScriptedChatModel(replies=REPLIES), path=path)
    again, result = play(model=ScriptedChatModel(replies=[]), path=path)  # any call would raise
    assert again.resumed and result == expected
    assert again.record == json.loads(json.dumps(first.record))
    assert again.game.get_step_rewards() == first.game.get_step_rewards()


def test_a_snapshot_of_another_game_is_refused(tmp_path):
    path = tmp_path / "game.json"
    play(model=ScriptedChatModel(replies=REPLIES), path=path)
    other = CopycatRPSGame(game_config={"seed": 1, "rounds": 4})
    with pytest.raises(ValueError):
        Runner(game=other, models={PLAYER: ScriptedChatModel(replies=[])}, checkpoint_path=path)


def test_a_replay_whose_prompt_differs_is_refused(tmp_path):
    path = tmp_path / "game.json"
    with pytest.raises(ModelCallFailed):
        play(model=CrashingChatModel(replies=REPLIES, fail_at=2), path=path)
    snapshot = json.loads(path.read_text())
    snapshot["journal"][0]["messages"][1]["content"] += " (edited)"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(RuntimeError, match="differs"):
        play(model=ScriptedChatModel(replies=REPLIES[2:]), path=path)


def test_snapshots_are_written_atomically(tmp_path):
    path = tmp_path / "game.json"
    play(model=ScriptedChatModel(replies=REPLIES), path=path)
    assert [p.name for p in tmp_path.iterdir()] == ["game.json"]


def test_checkpoint_paths_are_named_by_the_whole_game_config(tmp_path):
    a = checkpoint_path(checkpoint_dir=tmp_path, game="copycat_rps", game_config={"seed": 0, "rounds": 4})
    b = checkpoint_path(checkpoint_dir=tmp_path, game="copycat_rps", game_config={"rounds": 4, "seed": 0})
    c = checkpoint_path(checkpoint_dir=tmp_path, game="copycat_rps", game_config={"seed": 0, "rounds": 5})
    d = checkpoint_path(checkpoint_dir=tmp_path, game="copycat_rps", game_config={"seed": 1, "rounds": 4})
    assert a == b  # key order never matters
    assert len({a, c, d}) == 3
    assert (tmp_path / f"meta_{a.stem}.yaml").exists()
