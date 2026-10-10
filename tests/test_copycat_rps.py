import json

import pytest

from self_play.game import THINK, DialogueMode, Game
from self_play.games import GAMES
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.copycat_rps.game import CHOICES, PLAYER


def new_game(*, seed=0, rounds=4):
    return CopycatRPSGame(game_config={"seed": seed, "rounds": rounds})


def think(*, game):
    assert game.get_dialogue_phase().mode is DialogueMode.THINKING
    assert game.step(player_id=PLAYER, action=THINK).accepted


def play(*, game, moves):
    """Each move after the thinking turn that precedes it."""
    for move in moves:
        think(game=game)
        assert game.step(player_id=PLAYER, action=move).accepted


def test_registered_by_name():
    assert GAMES["copycat_rps"] is CopycatRPSGame


def test_opening_move_comes_from_the_seed():
    assert {new_game(seed=seed).state["opening_move"] for seed in range(30)} == set(CHOICES)
    assert new_game(seed=7).state == new_game(seed=7).state


def test_opponent_repeats_the_players_previous_move():
    game = new_game(rounds=4)
    opening = game.state["opening_move"]
    play(game=game, moves=["rock", "paper", "paper", "scissors"])
    assert [entry["opponent"] for entry in game.state["history"]] == [opening, "rock", "paper", "paper"]


def test_exploiting_the_copycat_wins_every_round_after_the_first():
    game = new_game(rounds=5)
    play(game=game, moves=["rock", "paper", "scissors", "rock", "paper"])  # each move beats the previous one
    assert [entry["winner"] for entry in game.state["history"][1:]] == [PLAYER] * 4
    assert game.get_outcome()["done"] and game.get_outcome()["rewards"] == {PLAYER: 1.0}
    assert game.get_dialogue_phase() is None


def test_rewards_zero_until_done_and_draw_is_half():
    game = new_game(rounds=2)
    assert game.get_outcome()["rewards"] == {PLAYER: 0.0}
    game.state["opening_move"] = "rock"
    play(game=game, moves=["rock", "rock"])  # draw, then the copycat plays rock again: draw
    assert game.get_outcome()["rewards"] == {PLAYER: 0.5}


# --- thinking ---------------------------------------------------------------------------


def test_the_first_action_is_always_thinking():
    game = new_game()
    phase = game.get_dialogue_phase()
    assert (phase.phase, phase.mode, phase.eligible) == ("thinking", DialogueMode.THINKING, (PLAYER,))
    assert game.get_legal_actions(player_id=PLAYER) == [THINK]
    assert not game.step(player_id=PLAYER, action="rock").accepted  # no move before thinking
    think(game=game)
    assert game.get_dialogue_phase().phase == "boast"
    assert game.get_legal_actions(player_id=PLAYER) == CHOICES
    assert not game.step(player_id=PLAYER, action=THINK).accepted  # no thinking out of turn


def test_a_thinking_turn_follows_every_round_but_not_the_last():
    game = new_game(rounds=3)
    phases = []
    for move in ["rock", "paper", "scissors"]:
        phases.append(game.get_dialogue_phase())
        think(game=game)
        assert game.step(player_id=PLAYER, action=move).accepted
    assert game.get_dialogue_phase() is None
    assert game.get_legal_actions(player_id=PLAYER) == []
    assert [phase.phase_id for phase in phases] == [-1, -2, -3]  # each thinking phase is new


def test_thinking_turns_are_steps_and_change_no_game_state():
    game = new_game(rounds=3)
    before = game.state
    think(game=game)
    assert game.state == before
    assert game.step(player_id=PLAYER, action="paper").accepted
    assert game.steps == [
        {"player_id": PLAYER, "action": THINK, "phase_id": -1},
        {"player_id": PLAYER, "action": "paper", "phase_id": 0},
    ]
    assert game.n_steps == 2


# --- step rewards -----------------------------------------------------------------------


def test_step_rewards_are_decided_at_the_end_rounds_scored_thinking_zero():
    game = new_game(rounds=4)
    game.state["opening_move"] = "paper"
    play(game=game, moves=["rock", "rock", "paper", "paper"])  # lose, draw, win, draw
    assert game.get_step_rewards() == [0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5]
    assert len(game.get_step_rewards()) == len(game.steps)


def test_step_rewards_are_refused_before_the_game_is_over():
    game = new_game(rounds=2)
    play(game=game, moves=["rock"])
    with pytest.raises(RuntimeError):
        game.get_step_rewards()


def test_default_step_rewards_give_every_step_the_final_reward():
    game = new_game(rounds=3)
    game.state["opening_move"] = "scissors"
    play(game=game, moves=["rock", "paper", "scissors"])  # win, win, win
    assert Game.get_step_rewards(game) == [1.0] * 6


# --- phases and the opponent's lines ---------------------------------------------------


def test_phases_cycle_through_the_four_modes_with_the_opponents_lines():
    game = new_game(rounds=8)
    seen = []
    for _ in range(8):
        think(game=game)
        phase = game.get_dialogue_phase()
        seen.append((phase.phase, phase.mode, game.get_observation(player_id=PLAYER)["opponent_says"]))
        assert game.step(player_id=PLAYER, action="rock").accepted
    expected_modes = [DialogueMode.ACTION, DialogueMode.RESPONSE_WINDOW, DialogueMode.DISCUSSION, DialogueMode.SIMULTANEOUS] * 2
    assert [mode for _, mode, _ in seen] == expected_modes
    assert [name for name, _, _ in seen[:4]] == ["boast", "comment", "chat", "claim"]
    assert seen[0][2] == "I'm sure I'll beat you with this"
    assert seen[1][2] == "The player is doing whatever they want"
    assert seen[2][2] == "I think I'm good at this"
    assert seen[3][2].startswith("I will play ")


def claims(*, seeds, moves):
    """(claimed move, real move) for round 4 of each seed, after playing `moves` for rounds 1-3."""
    out = []
    for seed in seeds:
        game = new_game(seed=seed, rounds=4)
        play(game=game, moves=moves)
        claimed = game.get_observation(player_id=PLAYER)["opponent_says"].removeprefix("I will play ").rstrip(".")
        play(game=game, moves=["rock"])
        out.append((claimed, game.state["history"][-1]["opponent"]))
    return out


def test_the_claim_is_true_or_a_lie_on_a_fair_coin():
    results = claims(seeds=range(200), moves=["rock", "paper", "scissors"])
    truthful = sum(claimed == real for claimed, real in results)
    assert 70 < truthful < 130
    assert all(claimed in CHOICES for claimed, _ in results)
    assert all(real == "scissors" for _, real in results)  # the claim never changes the copycat's move


def test_the_claim_is_fixed_by_the_seed():
    assert claims(seeds=[5], moves=["rock"] * 3) == claims(seeds=[5], moves=["rock"] * 3)


# --- legality and parsing ---------------------------------------------------------------


def test_illegal_actions_change_nothing():
    game = new_game(rounds=1)
    think(game=game)
    before = game.to_dict()
    for player_id, action in [(PLAYER, "lizard"), ("opponent", "rock"), (PLAYER, "Rock"), (PLAYER, THINK)]:
        result = game.step(player_id=player_id, action=action)
        assert not result.accepted and result.error
    assert game.to_dict() == before
    assert game.step(player_id=PLAYER, action="rock").accepted
    assert not game.step(player_id=PLAYER, action="rock").accepted  # the game is over
    assert game.n_steps == 2


def test_parse_answer_reads_the_move():
    game = new_game()
    assert game.parse_answer(player_id=PLAYER, answer=" Paper. ") == "paper"
    assert game.parse_answer(player_id=PLAYER, answer="**rock**") == "rock"
    assert game.parse_answer(player_id=PLAYER, answer="lizard") == "lizard"  # readable, so step rejects it
    with pytest.raises(ValueError):
        game.parse_answer(player_id=PLAYER, answer="  ")


# --- what the player sees ---------------------------------------------------------------


def test_round_one_observation_and_prompt_do_not_depend_on_the_opening_move():
    seen = set()
    for opening in CHOICES:
        game = new_game(rounds=2)
        game.state["opening_move"] = opening
        for _ in range(2):  # during the opening thinking turn, then at the round 1 move
            observation = game.get_observation(player_id=PLAYER)
            seen.add((json.dumps(observation), game.render_observation(observation=observation)))
            if game.get_dialogue_phase().mode is DialogueMode.THINKING:
                think(game=game)
    assert len(seen) == 2


def test_observation_shows_only_played_rounds():
    game = new_game(rounds=4)
    for round_number in range(1, 5):
        think(game=game)
        observation = game.get_observation(player_id=PLAYER)
        assert set(observation) == {"round", "rounds", "phase", "opponent_says", "history", "score"}
        assert observation["round"] == round_number
        assert len(observation["history"]) == round_number - 1  # this round's moves are not in it yet
        assert game.step(player_id=PLAYER, action="rock").accepted


def test_prompt_starts_with_the_rules_and_never_reveals_the_policy():
    game = new_game(rounds=3)
    observation = game.get_observation(player_id=PLAYER)
    rules = game.get_rules(observation=observation)
    assert game.render_observation(observation=observation).startswith(rules)
    assert "3 rounds" in rules
    assert not any(word in rules.lower() for word in ("repeat", "copy", "previous"))


def test_the_move_menu_is_only_in_the_answer_request():
    game = new_game()
    observation = game.get_observation(player_id=PLAYER)
    assert "Your answer" not in game.render_observation(observation=observation)
    request = game.render_answer_request(legal_actions=CHOICES)
    assert request == "Your answer is your move for this round: one of rock, paper, scissors."


# --- checkpoints ------------------------------------------------------------------------


def test_save_round_trip_through_json_continues_identically():
    game = new_game(seed=3, rounds=4)
    play(game=game, moves=["paper", "scissors"])  # saved while a thinking turn is owed
    restored = CopycatRPSGame.from_dict(data=json.loads(json.dumps(game.to_dict())))
    assert restored.to_dict() == game.to_dict()
    assert restored.get_dialogue_phase() == game.get_dialogue_phase()
    assert restored.get_observation(player_id=PLAYER) == game.get_observation(player_id=PLAYER)
    play(game=game, moves=["rock", "rock"])
    play(game=restored, moves=["rock", "rock"])
    assert restored.to_dict() == game.to_dict()


def test_from_dict_rejects_another_game():
    data = new_game().to_dict()
    with pytest.raises(ValueError):
        CopycatRPSGame.from_dict(data={**data, "game": "coup"})
