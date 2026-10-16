import asyncio
import json
from collections import Counter

import pytest

from self_play.chat_model import ScriptedChatModel
from self_play.game import THINK, DialogueMode
from self_play.games import GAMES
from self_play.games.coup import CoupGame
from self_play.games.coup.game import CARDS, COPIES
from self_play.runner import Runner


def new_game(*, seed=0, **settings):
    return CoupGame(game_config={"seed": seed, "players": 2, **settings})


def think_all(*, game):
    """Play every owed thinking turn."""
    while (phase := game.get_dialogue_phase()) is not None and phase.mode is DialogueMode.THINKING:
        for player_id in phase.eligible:
            assert game.step(player_id=player_id, action=THINK).accepted


def deciding(*, game):
    think_all(game=game)
    return game.get_dialogue_phase().eligible[0]


def all_cards(*, game):
    state = game.state
    return Counter(state["deck"] + [c for pid in game.player_ids for c in state["hands"][pid] + state["revealed"][pid]])


def reply(*, answer):
    return f"<thinking>scratch</thinking>\n<answer>{answer}</answer>\n[STOP]"


def test_registered_by_name():
    assert GAMES["coup"] is CoupGame


# --- setup ------------------------------------------------------------------------------


def test_the_deal_comes_from_the_seed():
    assert new_game(seed=4).state == new_game(seed=4).state
    assert len({json.dumps(new_game(seed=seed).state["hands"]) for seed in range(20)}) > 1
    assert {new_game(seed=seed).state["actor"] for seed in range(20)} == {"p1", "p2"}


def test_setup_two_cards_each_fifteen_in_all_and_the_starter_gets_one_coin():
    for seed in range(20):
        game = new_game(seed=seed)
        state = game.state
        assert game.player_ids == ["p1", "p2"]
        assert all(len(state["hands"][pid]) == 2 for pid in game.player_ids)
        assert len(state["deck"]) == 11
        assert all_cards(game=game) == Counter({card: COPIES for card in CARDS})
        starter = state["actor"]
        other = next(pid for pid in game.player_ids if pid != starter)
        assert state["coins"] == {starter: 1, other: 2}


def test_only_two_players_for_now():
    for players in (1, 3, 6):
        with pytest.raises(ValueError):
            new_game(players=players)


# --- legal actions ----------------------------------------------------------------------


def test_everyone_thinks_first_then_only_the_actor_may_act():
    game = new_game()
    phase = game.get_dialogue_phase()
    assert (phase.mode, phase.eligible) == (DialogueMode.THINKING, ("p1", "p2"))
    assert game.get_legal_actions(player_id="p1") == [THINK]
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    phase = game.get_dialogue_phase()
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("action", DialogueMode.ACTION, (actor,), 1)
    assert game.get_legal_actions(player_id=other) == []


def test_legal_actions_match_the_design():
    game = new_game()
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    income, foreign_aid, coup = {"action": "income"}, {"action": "foreign_aid"}, {"action": "coup", "target": other}
    for coins, expected in [(0, [income, foreign_aid]), (6, [income, foreign_aid]), (7, [income, foreign_aid, coup]),
                            (9, [income, foreign_aid, coup]), (10, [coup]), (12, [coup])]:
        game.state["coins"][actor] = coins
        assert game.get_legal_actions(player_id=actor) == expected


def test_lose_influence_offers_each_distinct_card_once():
    game = new_game()
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    game.state["coins"][actor] = 7
    game.state["hands"][other] = ["duke", "captain"]
    assert game.step(player_id=actor, action={"action": "coup", "target": other}).accepted
    phase = game.get_dialogue_phase()
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("lose_influence", DialogueMode.ACTION, (other,), 0)
    assert game.get_legal_actions(player_id=other) == [{"reveal": "duke"}, {"reveal": "captain"}]
    assert game.get_legal_actions(player_id=actor) == []
    game.state["hands"][other] = ["contessa", "contessa"]
    assert game.get_legal_actions(player_id=other) == [{"reveal": "contessa"}]


def test_illegal_actions_change_nothing():
    game = new_game()
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    before = game.to_dict()
    for player_id, action in [(actor, {"action": "coup", "target": other}), (other, {"action": "income"}),
                              (actor, {"action": "tax"}), (actor, "income"), (actor, THINK)]:
        result = game.step(player_id=player_id, action=action)
        assert not result.accepted and result.error
    assert game.to_dict() == before


# --- transitions ------------------------------------------------------------------------


def test_income_foreign_aid_and_coup_change_coins_and_pass_the_turn():
    game = new_game()
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    start = dict(game.state["coins"])
    assert game.step(player_id=actor, action={"action": "income"}).accepted
    assert game.state["coins"][actor] == start[actor] + 1
    assert game.get_dialogue_phase().mode is DialogueMode.THINKING
    assert game.get_dialogue_phase().eligible == (other,)  # only the next actor thinks
    assert deciding(game=game) == other
    assert game.step(player_id=other, action={"action": "foreign_aid"}).accepted
    assert game.state["coins"][other] == start[other] + 2
    assert deciding(game=game) == actor
    game.state["coins"][actor] = 8
    events = game.step(player_id=actor, action={"action": "coup", "target": other}).public_events
    assert game.state["coins"][actor] == 1 and events
    assert game.get_dialogue_phase().phase == "lose_influence"


def test_the_loser_chooses_the_card_and_elimination_ends_the_game():
    game = new_game()
    actor = deciding(game=game)
    other = "p2" if actor == "p1" else "p1"
    game.state["hands"][other] = ["duke", "captain"]
    game.state["coins"][actor] = 14
    assert game.step(player_id=actor, action={"action": "coup", "target": other}).accepted
    assert game.step(player_id=other, action={"reveal": "captain"}).accepted
    assert game.state["hands"][other] == ["duke"] and game.state["revealed"][other] == ["captain"]
    assert deciding(game=game) == other
    assert game.step(player_id=other, action={"action": "income"}).accepted
    assert deciding(game=game) == actor
    assert game.step(player_id=actor, action={"action": "coup", "target": other}).accepted
    assert game.step(player_id=other, action={"reveal": "duke"}).accepted
    assert game.get_dialogue_phase() is None
    assert game.state["coins"][other] == 0  # coins return to the Treasury
    assert game.get_outcome() == {"done": True, "rewards": {actor: 1.0, other: 0.0}, "winner": actor,
                                  "termination_reason": "last_player_standing"}
    assert all_cards(game=game) == Counter({card: COPIES for card in CARDS})
    assert len(game.get_step_rewards()) == len(game.steps)


def test_rule_based_games_from_the_seed_always_end_with_one_winner():
    for seed in range(30):
        game = new_game(seed=seed)
        while game.get_dialogue_phase() is not None:
            player_id = deciding(game=game)
            legal = game.get_legal_actions(player_id=player_id)
            coups = [action for action in legal if action.get("action") == "coup"]
            assert game.step(player_id=player_id, action=coups[0] if coups else legal[0]).accepted
        outcome = game.get_outcome()
        assert outcome["done"] and outcome["termination_reason"] == "last_player_standing"
        assert sorted(outcome["rewards"].values()) == [0.0, 1.0]
        assert all(coins >= 0 for coins in game.state["coins"].values())


def test_the_turn_limit_ends_the_game_without_a_winner():
    game = new_game(max_turns=3)
    for _ in range(3):
        player_id = deciding(game=game)
        assert game.step(player_id=player_id, action={"action": "income"}).accepted
    assert game.get_dialogue_phase() is None
    assert game.get_outcome() == {"done": True, "rewards": {"p1": 0.0, "p2": 0.0}, "winner": None,
                                  "termination_reason": "turn_limit"}


# --- hidden information -----------------------------------------------------------------


def seen_by(*, game, player_id):
    observation = game.get_observation(player_id=player_id)
    return json.dumps(observation), game.render_observation(observation=observation)


def test_no_opponent_card_appears_in_an_observation_or_prompt():
    """Swapping the opponent's face-down cards (and the deck) changes nothing a player sees,
    at the opening, in the action phase and while revealing a card."""
    for seed in range(10):
        game = new_game(seed=seed)
        checkpoints = []
        checkpoints.append(game.to_dict())  # opening thinking
        actor = deciding(game=game)
        other = "p2" if actor == "p1" else "p1"
        checkpoints.append(game.to_dict())  # action phase
        game.state["coins"][actor] = 7
        game.state["hands"][other] = ["duke", "captain"]
        assert game.step(player_id=actor, action={"action": "coup", "target": other}).accepted
        checkpoints.append(game.to_dict())  # lose_influence phase
        for data in checkpoints:
            for viewer, hidden in [(actor, other), (other, actor)]:
                real = CoupGame.from_dict(data=data)
                swapped = CoupGame.from_dict(data=data)
                swapped.state["hands"][hidden] = ["ambassador"] * len(real.state["hands"][hidden])
                swapped.state["deck"] = list(reversed(real.state["deck"]))
                assert seen_by(game=real, player_id=viewer) == seen_by(game=swapped, player_id=viewer)
                assert real.get_observation(player_id=viewer)["your_cards"] == real.state["hands"][viewer]


def test_the_observation_has_exactly_the_designed_fields():
    game = new_game()
    observation = game.get_observation(player_id="p1")
    assert set(observation) == {"you", "your_cards", "players", "deck_size", "turn", "max_turns", "actor",
                                "phase", "deciding", "log", "winner"}
    assert set(observation["players"][0]) == {"id", "coins", "hidden_cards", "revealed", "alive"}


def test_prompt_starts_with_the_rules_and_the_menu_is_only_in_the_answer_request():
    game = new_game()
    actor = deciding(game=game)
    observation = game.get_observation(player_id=actor)
    text = game.render_observation(observation=observation)
    assert text.startswith(game.get_rules(observation=observation))
    assert "Your answer" not in text
    request = game.render_answer_request(legal_actions=game.get_legal_actions(player_id=actor))
    assert '{"action": "income"}' in request and '{"action": "foreign_aid"}' in request


# --- parsing ----------------------------------------------------------------------------


def test_parse_answer_reads_json_actions():
    game = new_game()
    assert game.parse_answer(player_id="p1", answer='{"action": "income"}') == {"action": "income"}
    assert game.parse_answer(player_id="p1", answer='```json\n{"action": "Coup", "target": "P2"}\n```') == {
        "action": "coup", "target": "p2"}
    assert game.parse_answer(player_id="p1", answer=' {"reveal": "Duke"} ') == {"reveal": "duke"}
    assert game.parse_answer(player_id="p1", answer="Pass.") == "pass"
    for bad in ("", "   ", "take income please"):
        with pytest.raises(ValueError):
            game.parse_answer(player_id="p1", answer=bad)


# --- checkpoints and the runner ---------------------------------------------------------


def test_save_round_trip_through_json_continues_identically():
    game = new_game(seed=3)
    for _ in range(3):
        player_id = deciding(game=game)
        assert game.step(player_id=player_id, action={"action": "income"}).accepted
    restored = CoupGame.from_dict(data=json.loads(json.dumps(game.to_dict())))
    assert restored.to_dict() == game.to_dict()
    for g in (game, restored):
        player_id = deciding(game=g)
        assert g.step(player_id=player_id, action={"action": "foreign_aid"}).accepted
    assert restored.to_dict() == game.to_dict()


def test_a_scripted_two_player_game_ends_with_one_winner():
    """p1 starts with 14 coins, so its first Coup is forced (no call); p2 reveals Duke and
    takes Income; p1 chooses a second Coup at 7 coins; p2's last card is revealed (forced)."""
    game = new_game(seed=0, talk_rounds=0)
    game.state["actor"] = "p1"
    game.state["coins"] = {"p1": 14, "p2": 2}
    game.state["hands"]["p2"] = ["duke", "captain"]
    coup = reply(answer='{"action": "coup", "target": "p2"}')
    models = {
        "p1": ScriptedChatModel(replies=[reply(answer="plan"), reply(answer="plan again"), coup]),
        "p2": ScriptedChatModel(replies=[reply(answer="plan"), reply(answer='{"reveal": "duke"}'),
                                         reply(answer="plan again"), reply(answer='{"action": "income"}')]),
    }
    result = asyncio.run(Runner(game=game, models=models).run())
    assert result["outcome"]["done"] and result["outcome"]["winner"] == "p1"
    assert result["outcome"]["rewards"] == {"p1": 1.0, "p2": 0.0}
    assert result["invalid"] == {"p1": 0, "p2": 0}
    assert all(not model._replies for model in models.values())  # every scripted reply was used
    assert game.state["revealed"]["p2"] == ["duke", "captain"]
    assert len(result["step_rewards"]) == len(game.steps)
