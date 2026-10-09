import asyncio

from self_play.chat_model import ScriptedChatModel, strip_stop
from self_play_env.smoke_game import FALLBACK, parse_choice, play_smoke_game


def test_strip_stop_cuts_at_marker():
    assert strip_stop(text="thinking <answer>rock</answer> [STOP] trailing") == "thinking <answer>rock</answer>"
    assert strip_stop(text="  no marker  ") == "no marker"


def test_parse_choice_requires_one_valid_answer():
    assert parse_choice(reply="I pick <answer>Paper</answer>[STOP]") == "paper"
    assert parse_choice(reply="<answer>rock</answer> or <answer>paper</answer>") is None
    assert parse_choice(reply="<answer>lizard</answer>") is None
    assert parse_choice(reply="[STOP] <answer>rock</answer>") is None  # text after the marker is ignored


def test_game_scores_and_fallback():
    seat0 = ScriptedChatModel(replies=["<answer>paper</answer>", "<answer>paper</answer>", "garbage"])
    seat1 = ScriptedChatModel(replies=["<answer>rock</answer>", "<answer>scissors</answer>", "<answer>paper</answer>"])
    result = asyncio.run(play_smoke_game(models=[seat0, seat1], rounds=3))
    assert result.round_winners == [0, 1, 1]  # the invalid third reply falls back to rock
    assert result.seats[0].choices[2] == FALLBACK
    assert result.seats[0].invalid == 1
    assert result.scores() == [0.0, 1.0]


def test_each_call_is_stateless_and_hides_nothing_unplayed():
    seat0 = ScriptedChatModel(replies=["<answer>rock</answer>"] * 2)
    seat1 = ScriptedChatModel(replies=["<answer>paper</answer>"] * 2)
    asyncio.run(play_smoke_game(models=[seat0, seat1], rounds=2))
    for model in (seat0, seat1):
        assert all(len(messages) == 2 for messages in model.calls)  # system + one user message, never growing
    first_round_prompt = seat0.calls[0][1]["content"]
    assert "paper" not in first_round_prompt  # the opponent's simultaneous choice is never shown early
    assert "your opponent chose paper" in seat0.calls[1][1]["content"]
