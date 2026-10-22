import asyncio
from dataclasses import replace

import pytest

from self_play.chat_model import ChatModel, ModelCallFailed, ScriptedChatModel
from self_play.game import DialogueMode
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.copycat_rps.game import PLAYER
from self_play.prompting import TALK_INSTRUCTION, TALK_REQUIRED_INSTRUCTION, parse_reply
from self_play.runner import Runner

# Copycat over 2 rounds asks, in order: think, talk + answer (round 1 is ACTION with one
# talk round), think, answer (round 2 is a RESPONSE_WINDOW without talk).


def reply(*, answer, thinking="scratch"):
    return f"<thinking>{thinking}</thinking>\n<answer>{answer}</answer>\n[STOP] ignored"


def run_game(*, replies, rounds=2, opening="paper", max_retries=0):
    game = CopycatRPSGame(game_config={"seed": 0, "rounds": rounds})
    game.state["opening_move"] = opening
    model = ScriptedChatModel(replies=replies)
    runner = Runner(game=game, models={PLAYER: model}, max_retries=max_retries)
    result = asyncio.run(runner.run())
    return runner, model, result


def calls(*, runner):
    return [entry for entry in runner.record if entry["kind"] == "call"]


def prompt(*, model, index):
    return model.calls[index][1]["content"]


def test_a_full_game_with_valid_answers():
    runner, model, result = run_game(replies=[
        reply(answer="I expect paper."), reply(answer="SILENCE"), reply(answer="scissors"),
        reply(answer="NO CHANGE"), reply(answer="rock"),
    ])
    assert [entry["call"] for entry in calls(runner=runner)] == ["think", "talk", "answer", "think", "answer"]
    assert [entry["player"] for entry in runner.game.state["history"]] == ["scissors", "rock"]
    assert result["outcome"]["rewards"] == {PLAYER: 1.0}  # scissors beats paper, rock beats the copied scissors
    assert result["step_rewards"] == [0.0, 1.0, 0.0, 1.0]
    assert result["invalid"] == {PLAYER: 0}
    assert len(model.calls) == 5  # every call was used, none extra


def test_notes_come_only_from_thinking_turns():
    runner, model, _ = run_game(replies=[
        reply(answer="Opponent may copy me.", thinking="first think"),
        reply(answer="SILENCE", thinking="secret talk reasoning"),
        reply(answer="rock", thinking="secret move reasoning"),
        reply(answer="NO CHANGE"),
        reply(answer="paper"),
    ])
    assert "Your notes: (none yet)" in prompt(model=model, index=0)
    for index in (1, 2, 3, 4):
        assert "Your notes:\nOpponent may copy me." in prompt(model=model, index=index)
    assert runner.notes(player_id=PLAYER) == "Opponent may copy me."  # NO CHANGE kept them
    for index in range(5):
        assert "secret" not in prompt(model=model, index=index)  # regular thinking never comes back


def test_a_thinking_turn_replaces_the_notes_entirely():
    runner, model, _ = run_game(replies=[
        reply(answer="old notes"), reply(answer="SILENCE"), reply(answer="rock"),
        reply(answer="new notes"), reply(answer="paper"),
    ])
    assert "Your notes:\nnew notes" in prompt(model=model, index=4)
    assert "old notes" not in prompt(model=model, index=4)
    assert runner.notes(player_id=PLAYER) == "new notes"


def test_an_illegal_answer_falls_back_to_the_first_legal_action_and_counts_invalid():
    runner, _, result = run_game(replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="lizard"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert runner.game.state["history"][0]["player"] == "rock"
    assert result["invalid"] == {PLAYER: 1}
    attempts = [e for e in runner.record if e["kind"] == "action" and e["phase_id"] == 0]
    assert [(e["source"], e["accepted"]) for e in attempts] == [("model", False), ("fallback", True)]
    assert "not legal" in attempts[0]["error"]


def test_a_reply_without_an_answer_falls_back_and_counts_invalid():
    runner, _, result = run_game(replies=[
        reply(answer="notes"), reply(answer="SILENCE"), "I choose paper, obviously.",
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert runner.game.state["history"][0]["player"] == "rock"
    assert result["invalid"] == {PLAYER: 1}


def test_a_thinking_turn_without_an_answer_keeps_the_notes_and_counts_invalid():
    runner, _, result = run_game(replies=[
        reply(answer="keep me"), reply(answer="SILENCE"), reply(answer="rock"),
        "no tags at all", reply(answer="paper"),
    ])
    assert runner.notes(player_id=PLAYER) == "keep me"
    assert result["invalid"] == {PLAYER: 1}


def test_retries_show_the_reason_and_a_good_retry_is_not_invalid():
    runner, model, result = run_game(max_retries=1, replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="lizard"), reply(answer="scissors"),
        reply(answer="NO CHANGE"), reply(answer="rock"),
    ])
    assert "Your previous answer was rejected:" in prompt(model=model, index=3)
    assert "lizard" in prompt(model=model, index=3)
    assert runner.game.state["history"][0]["player"] == "scissors"
    assert result["invalid"] == {PLAYER: 0}


class FailingChatModel(ChatModel):
    async def complete(self, *, messages):
        raise ModelCallFailed("the seat's run ended")


def test_a_failed_model_call_crashes_the_game():
    game = CopycatRPSGame(game_config={"seed": 0, "rounds": 2})
    runner = Runner(game=game, models={PLAYER: FailingChatModel()})
    with pytest.raises(ModelCallFailed):
        asyncio.run(runner.run())


def test_talk_is_recorded_and_shown_unless_silent():
    runner, model, _ = run_game(replies=[
        reply(answer="notes"), reply(answer="Good luck!"), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    messages = [e for e in runner.record if e["kind"] == "message"]
    assert [(e["player_id"], e["content"]) for e in messages] == [(PLAYER, "Good luck!")]
    assert "player said: Good luck!" in prompt(model=model, index=2)
    assert "player said: Good luck!" in prompt(model=model, index=3)  # the next thinking turn sees it


def test_all_four_modes_run_including_simultaneous():
    replies = [reply(answer="notes")]
    for talk_rounds, move in [(1, "rock"), (0, "paper"), (1, "scissors"), (0, "rock")]:
        replies += [reply(answer="SILENCE")] * talk_rounds + [reply(answer=move)]
        replies.append(reply(answer="NO CHANGE"))
    replies.pop()  # no thinking turn after the last round
    runner, model, result = run_game(rounds=4, replies=replies)
    assert [entry["player"] for entry in runner.game.state["history"]] == ["rock", "paper", "scissors", "rock"]
    assert [entry["phase"] for entry in runner.game.state["history"]] == ["boast", "comment", "chat", "claim"]
    assert result["invalid"] == {PLAYER: 0}
    assert len(model.calls) == len(replies)


def test_every_call_is_linked_to_the_step_it_led_to():
    runner, _, _ = run_game(replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    # steps: 0 think, 1 round-1 move, 2 think, 3 round-2 move
    assert runner.call_step_ids() == [0, 1, 1, 2, 3]


def test_prompts_are_the_whole_context_and_hold_the_game_text():
    _, model, _ = run_game(replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    for messages in model.calls:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"].startswith("You are playing rock paper scissors")
    assert "Your answer is your move" in prompt(model=model, index=2)
    assert "Your answer is your move" not in prompt(model=model, index=0)  # thinking turns show no menu
    assert "NO CHANGE" in prompt(model=model, index=0)


def test_models_must_match_the_players():
    game = CopycatRPSGame(game_config={"seed": 0, "rounds": 1})
    with pytest.raises(ValueError):
        Runner(game=game, models={"someone_else": ScriptedChatModel(replies=[])})


def test_parse_reply_cuts_at_the_stop_string_and_never_guesses():
    assert parse_reply(reply="<thinking>t</thinking><answer>rock</answer>[STOP]<answer>paper</answer>") == ("t", "rock")
    assert parse_reply(reply="just rock") == (None, None)
    assert parse_reply(reply="[STOP]<answer>rock</answer>") == (None, None)


class RequiredTalkCopycat(CopycatRPSGame):
    """Copycat whose round-1 talk is required (the engine's only talk_required user in tests)."""

    def get_dialogue_phase(self):
        phase = super().get_dialogue_phase()
        if phase is not None and phase.mode is DialogueMode.ACTION:
            return replace(phase, talk_required=True)
        return phase


def run_required(*, replies, max_retries=0):
    game = RequiredTalkCopycat(game_config={"seed": 0, "rounds": 2})
    game.state["opening_move"] = "paper"
    model = ScriptedChatModel(replies=replies)
    runner = Runner(game=game, models={PLAYER: model}, max_retries=max_retries)
    result = asyncio.run(runner.run())
    return runner, model, result


def action_phase(*, runner):
    return next(e for e in runner.record if e["kind"] == "phase" and e["mode"] == "action")


def messages_of(*, runner):
    return [(e["player_id"], e["content"]) for e in runner.record if e["kind"] == "message"]


def test_required_talk_with_a_real_message_posts_it():
    runner, model, result = run_required(replies=[
        reply(answer="notes"), reply(answer="Good luck!"), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert messages_of(runner=runner) == [(PLAYER, "Good luck!")]
    assert TALK_REQUIRED_INSTRUCTION in prompt(model=model, index=1)
    assert "SILENCE" not in TALK_REQUIRED_INSTRUCTION.replace("or SILENCE is not accepted", "")
    assert TALK_INSTRUCTION not in prompt(model=model, index=1)
    assert result["invalid"] == {PLAYER: 0}
    assert action_phase(runner=runner)["talk_required"] is True


def test_required_talk_answered_with_silence_is_retried_then_counts_one_invalid_and_posts_nothing():
    runner, model, result = run_required(max_retries=2, replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer=""), reply(answer="silence."), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert "Your previous answer was rejected:" in prompt(model=model, index=2)
    assert "SILENCE is not accepted" in prompt(model=model, index=2)
    assert [e["call"] for e in calls(runner=runner)].count("talk") == 3  # first try plus max_retries retries
    assert messages_of(runner=runner) == []
    assert result["invalid"] == {PLAYER: 1}
    assert runner.game.state["history"][0]["player"] == "rock"  # the game went on


def test_required_talk_retry_that_says_something_is_not_invalid():
    runner, model, result = run_required(max_retries=1, replies=[
        reply(answer="notes"), "no tags here", reply(answer="Hello"), reply(answer="rock"),
        reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert "no <answer>...</answer> was found" in prompt(model=model, index=2)
    assert messages_of(runner=runner) == [(PLAYER, "Hello")]
    assert result["invalid"] == {PLAYER: 0}


def test_required_talk_without_retries_counts_invalid_at_once():
    runner, _, result = run_required(replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="rock"), reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert messages_of(runner=runner) == [] and result["invalid"] == {PLAYER: 1}


def test_a_normal_talk_phase_still_accepts_silence_and_counts_nothing():
    runner, model, result = run_game(max_retries=2, replies=[
        reply(answer="notes"), reply(answer="SILENCE"), reply(answer="rock"), reply(answer="NO CHANGE"), reply(answer="paper"),
    ])
    assert TALK_INSTRUCTION in prompt(model=model, index=1)
    assert messages_of(runner=runner) == [] and result["invalid"] == {PLAYER: 0}
    assert len(model.calls) == 5  # no retry was spent on the silence
    assert action_phase(runner=runner)["talk_required"] is False
