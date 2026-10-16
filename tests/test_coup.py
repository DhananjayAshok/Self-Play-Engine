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


def new_game(*, seed=0, players=2, **settings):
    return CoupGame(game_config={"seed": seed, "players": players, **settings})


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


def test_two_to_six_players_start_with_two_coins_each():
    for players in range(3, 7):
        for seed in range(10):
            game = new_game(seed=seed, players=players)
            assert game.player_ids == [f"p{i}" for i in range(1, players + 1)]
            assert game.state["coins"] == {pid: 2 for pid in game.player_ids}  # the 1-coin rule is 2 players only
            assert all(len(game.state["hands"][pid]) == 2 for pid in game.player_ids)
            assert all_cards(game=game) == Counter({card: COPIES for card in CARDS})
    for players in (0, 1, 7, "3"):
        with pytest.raises(ValueError):
            new_game(players=players)


def test_a_six_player_game_deals_twelve_cards_and_leaves_three():
    game = new_game(players=6)
    assert sum(len(hand) for hand in game.state["hands"].values()) == 12
    assert len(game.state["deck"]) == 3
    assert game.get_observation(player_id="p1")["deck_size"] == 3


def test_the_starting_player_comes_from_the_seed_for_every_player_count():
    for players in range(2, 7):
        starters = {new_game(seed=seed, players=players).state["actor"] for seed in range(60)}
        assert starters == {f"p{i}" for i in range(1, players + 1)}


def test_settings_are_checked():
    for settings in ({"thinking": "some"}, {"window_talk_rounds": -1}, {"window_talk_rounds": 1.5},
                     {"talk_rounds": -1}, {"max_turns": 0}, {"window_talk_rounds": True}):
        with pytest.raises(ValueError):
            new_game(**settings)
    for settings in ({"thinking": "actor"}, {"thinking": "all"}, {"window_talk_rounds": 0}, {"window_talk_rounds": 2}):
        new_game(**settings)  # accepted


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
    assert len(game.get_step_rewards()) == len(game.steps)


def test_rule_based_games_from_the_seed_always_end_with_one_winner():
    """Coup the first listed target whenever possible, else take Foreign Aid; reveal the first
    card. The turn limit is raised so that only elimination can end these games."""
    for players in range(2, 7):
        for seed in range(20):
            game = new_game(seed=seed, players=players, max_turns=1000)
            while game.get_dialogue_phase() is not None:
                player_id = deciding(game=game)
                assert game.alive(player_id=player_id)  # the eliminated never act
                legal = game.get_legal_actions(player_id=player_id)
                coups = [action for action in legal if action.get("action") == "coup"]
                assert all(game.alive(player_id=c["target"]) and c["target"] != player_id for c in coups)
                choice = coups[0] if coups else legal[1] if legal[0] == {"action": "income"} else legal[0]
                assert game.step(player_id=player_id, action=choice).accepted
            outcome = game.get_outcome()
            assert outcome["done"] and outcome["termination_reason"] == "last_player_standing"
            assert sorted(outcome["rewards"].values()) == [0.0] * (players - 1) + [1.0]
            assert all(coins >= 0 for coins in game.state["coins"].values())
            assert all_cards(game=game) == Counter({card: COPIES for card in CARDS})


def test_turns_go_clockwise_from_the_starter_and_skip_the_eliminated():
    game = new_game(players=4)
    game.state["actor"] = "p3"
    order = []
    for _ in range(4):
        player_id = deciding(game=game)
        order.append(player_id)
        assert game.step(player_id=player_id, action={"action": "income"}).accepted
    assert order == ["p3", "p4", "p1", "p2"]
    game.state["hands"]["p4"] = ["duke"]
    game.state["coins"]["p3"] = 7
    assert deciding(game=game) == "p3"
    assert game.step(player_id="p3", action={"action": "coup", "target": "p4"}).accepted
    assert game.step(player_id="p4", action={"reveal": "duke"}).accepted  # p4's last card
    assert not game.alive(player_id="p4") and game.state["coins"]["p4"] == 0
    assert deciding(game=game) == "p1"  # p4 is skipped
    game.state["coins"]["p1"] = 7
    coups = [a for a in game.get_legal_actions(player_id="p1") if a["action"] == "coup"]
    assert coups == [{"action": "coup", "target": "p2"}, {"action": "coup", "target": "p3"}]  # never p4


# --- thinking ---------------------------------------------------------------------------


def test_thinking_actor_asks_only_the_next_actor_and_all_asks_every_living_player():
    for thinking, expected in [("actor", ("p2",)), ("all", ("p1", "p2", "p4"))]:
        game = new_game(players=4, thinking=thinking)
        game.state["actor"] = "p1"
        game.state["hands"]["p3"] = ["duke"]
        game.state["coins"]["p1"] = 7
        phase = game.get_dialogue_phase()
        assert phase.mode is DialogueMode.THINKING and phase.eligible == ("p1", "p2", "p3", "p4")  # the opening
        think_all(game=game)
        assert game.step(player_id="p1", action={"action": "coup", "target": "p3"}).accepted
        assert game.step(player_id="p3", action={"reveal": "duke"}).accepted  # p3 is out
        phase = game.get_dialogue_phase()
        assert phase.mode is DialogueMode.THINKING and phase.eligible == expected


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


def coup_reply(*, target):
    return reply(answer=json.dumps({"action": "coup", "target": target}))


def run_scripted(*, game, replies):
    models = {pid: ScriptedChatModel(replies=replies[pid]) for pid in game.player_ids}
    runner = Runner(game=game, models=models)
    result = asyncio.run(runner.run())
    assert all(not model._replies for model in models.values())  # every scripted reply was used
    assert result["invalid"] == {pid: 0 for pid in game.player_ids}
    return result, runner, models


def calls_after_elimination(*, runner, player_id):
    """Model calls made to the player after the step in which it revealed its last card."""
    out_at = max(i for i, e in enumerate(runner.record) if e["kind"] == "action" and e["player_id"] == player_id)
    return [e for e in runner.record[out_at:] if e["kind"] == "call" and e["player_id"] == player_id]


def test_a_scripted_three_player_game_ends_with_one_winner_and_the_eliminated_are_never_asked():
    """p1 coups p3 (asked: two targets); p3 reveals Duke; p2 coups p3, whose last card is forced;
    then p2 and p1 alternate (p3 skipped) until p1's second chosen Coup removes p2."""
    game = new_game(players=3, talk_rounds=0)
    game.state["actor"] = "p1"
    game.state["coins"] = {"p1": 21, "p2": 7, "p3": 0}
    game.state["hands"] |= {"p2": ["contessa", "assassin"], "p3": ["duke", "captain"]}
    think = reply(answer="plan")
    replies = {
        # opening think; T1 coup p3; T3 think (its coup is forced: 14 coins, one target); T5 think, coup p2
        "p1": [think, coup_reply(target="p3"), think, think, coup_reply(target="p2")],
        # opening think; T2 think, coup p3; T3 reveal Contessa; T4 think, Income
        "p2": [think, think, coup_reply(target="p3"), reply(answer='{"reveal": "contessa"}'), think,
               reply(answer='{"action": "income"}')],
        # opening think; T1 reveal Duke; T2 its last card is forced; never asked again
        "p3": [think, reply(answer='{"reveal": "duke"}')],
    }
    result, runner, models = run_scripted(game=game, replies=replies)
    assert result["outcome"]["winner"] == "p1"
    assert result["outcome"]["rewards"] == {"p1": 1.0, "p2": 0.0, "p3": 0.0}
    assert len(models["p3"].calls) == 2
    for loser in ("p2", "p3"):
        assert calls_after_elimination(runner=runner, player_id=loser) == []
    actors = [step["player_id"] for step in game.steps if isinstance(step["action"], dict) and "action" in step["action"]]
    assert actors == ["p1", "p2", "p1", "p2", "p1"]


def test_a_scripted_six_player_game_ends_with_one_winner_and_the_eliminated_are_never_asked():
    """Everyone holds one card. p1 (100 coins) coups p2, p3, ... in turn; each other living
    player takes Income on its turn until it is removed. The last Coup has one target: forced."""
    game = new_game(players=6, talk_rounds=0)
    game.state["actor"] = "p1"
    game.state["coins"] = {pid: 100 if pid == "p1" else 0 for pid in game.player_ids}
    game.state["hands"] = {pid: ["duke"] for pid in game.player_ids}
    think, income = reply(answer="plan"), reply(answer='{"action": "income"}')
    replies = {"p1": [think]}
    for k in range(2, 6):  # rounds 1-4: p1 thinks (not before round 1) and chooses its Coup
        replies["p1"] += ([think] if k > 2 else []) + [coup_reply(target=f"p{k}")]
    replies["p1"] += [think]  # round 5: think, then the forced Coup on p6
    for k in range(2, 7):  # p_k plays rounds 1 .. k-2 and is removed in round k-1, before its turn
        replies[f"p{k}"] = [think] + [think, income] * (k - 2)
    result, runner, models = run_scripted(game=game, replies=replies)
    assert result["outcome"]["winner"] == "p1"
    assert result["outcome"]["rewards"] == {pid: 1.0 if pid == "p1" else 0.0 for pid in game.player_ids}
    for k in range(2, 7):
        assert calls_after_elimination(runner=runner, player_id=f"p{k}") == []
        assert len(models[f"p{k}"].calls) == 1 + 2 * (k - 2)
