import asyncio
import json
import random
from collections import Counter

import pytest

from self_play.chat_model import ChatModel
from self_play.game import THINK, DialogueMode
from self_play.games import GAMES
from self_play.games.sheriff import SheriffGame
from self_play.games.sheriff.game import COUNTS, HAND_SIZE, SHERIFF_TURNS, START_GOLD, sort_goods
from self_play.prompting import THINK_INSTRUCTION
from self_play.runner import Runner


def new_game(*, seed=0, players=3, **settings):
    return SheriffGame(game_config={"seed": seed, "players": players, **settings})


def think_all(*, game):
    """Play every owed thinking turn."""
    while (phase := game.get_dialogue_phase()) is not None and phase.mode is DialogueMode.THINKING:
        for player_id in phase.eligible:
            assert game.step(player_id=player_id, action=THINK).accepted


def current(*, game):
    think_all(game=game)
    return game.get_dialogue_phase()


def act(*, game, moves):
    """Apply (player, action) pairs in order; every one must be accepted."""
    for player_id, action in moves:
        result = game.step(player_id=player_id, action=action)
        assert result.accepted, (player_id, action, result.error)


def full_deck(*, players):
    return Counter({good: copies for good, copies in COUNTS[min(players, 4)].items() if copies})


def all_cards(*, game):
    state = game.state
    cards = state["deck"] + state["discards"][0] + state["discards"][1] + state["market"]["set_aside"]
    for pid in game.player_ids:
        cards += state["hands"][pid] + state["bags"][pid] + state["stand_legal"][pid] + state["stand_contraband"][pid]
    return Counter(cards)


def set_sheriff(*, game, sheriff):
    """Make `sheriff` this round's Sheriff, before anyone has acted in the Market."""
    game.state["sheriff"] = sheriff
    game.state["sheriffs"] = [sheriff]
    game.state["market"] = game._new_market(merchants=game.merchants())


def swap_cards(*, game, hands):
    """Give these players these hands, trading cards with the draw pile so cards are conserved."""
    for pid, hand in hands.items():
        game.state["deck"] += game.state["hands"][pid]
        need = Counter(hand)
        for good, count in need.items():
            for _ in range(count):
                game.state["deck"].remove(good)
        game.state["hands"][pid] = list(hand)


def table(*, players=3, sheriff="p1", hands=None, **settings):
    """A game with this Sheriff and these hands, after the opening thinking."""
    game = new_game(players=players, talk_rounds=0, **settings)
    set_sheriff(game=game, sheriff=sheriff)
    swap_cards(game=game, hands=hands or {})
    think_all(game=game)
    return game


def skip_market(*, game):
    """Every merchant keeps its hand."""
    while current(game=game).phase == "set_aside":
        act(game=game, moves=[(current(game=game).eligible[0], "done")])


def load_bags(*, game, bags):
    """Load these bags one card at a time, all merchants at once; a merchant whose cards are
    all in answers "done" (a full bag closes itself)."""
    loaded = {pid: 0 for pid in bags}
    while (phase := current(game=game)).phase == "load":
        for pid in phase.eligible:
            card = bags[pid][loaded[pid]] if loaded[pid] < len(bags[pid]) else None
            act(game=game, moves=[(pid, {"load": card} if card else "done")])
            loaded[pid] += 1


def declare_all(*, game, goods):
    while current(game=game).phase == "declare":
        pid = current(game=game).eligible[0]
        act(game=game, moves=[(pid, {"declare": goods[pid]})])


def play(*, game, choose, check=None):
    """Play to the end; `choose(legal)` picks each action, `check()` runs after every step."""
    while (phase := game.get_dialogue_phase()) is not None:
        assert phase.eligible
        for player_id in phase.eligible:
            legal = game.get_legal_actions(player_id=player_id)
            assert legal and len(legal) == len({json.dumps(option) for option in legal})
            assert game.step(player_id=player_id, action=choose(legal)).accepted
            if check:
                check()


def reply(*, answer):
    return f"<thinking>scratch</thinking>\n<answer>{answer}</answer>\n[STOP]"


def test_registered_by_name():
    assert GAMES["sheriff"] is SheriffGame


# --- setup ------------------------------------------------------------------------------


def test_setup_matches_the_rulebook_for_three_four_and_five_players():
    for players, total, rounds in [(3, 156, 9), (4, 204, 8), (5, 204, 10)]:
        assert sum(full_deck(players=players).values()) == total
        for seed in range(5):
            game = new_game(seed=seed, players=players)
            state = game.state
            assert game.player_ids == [f"p{i}" for i in range(1, players + 1)]
            assert all_cards(game=game) == full_deck(players=players)
            assert all(len(state["hands"][pid]) == HAND_SIZE for pid in game.player_ids)
            assert [len(pile) for pile in state["discards"]] == [5, 5]
            assert len(state["deck"]) == total - HAND_SIZE * players - 10
            assert state["gold"] == {pid: START_GOLD for pid in game.player_ids}
            assert state["rounds"] == rounds == SHERIFF_TURNS[players] * players
    assert "bread" not in all_cards(game=new_game(players=3))  # the 3-player deck has no 4+ cards


def test_the_deal_and_the_first_sheriff_come_from_the_seed():
    assert new_game(seed=4).state == new_game(seed=4).state
    assert len({json.dumps(new_game(seed=seed).state["hands"]) for seed in range(20)}) > 1
    for players in (3, 4, 5):
        sheriffs = {new_game(seed=seed, players=players).state["sheriff"] for seed in range(60)}
        assert sheriffs == {f"p{i}" for i in range(1, players + 1)}


def test_player_count_and_settings_are_checked():
    for players in (2, 6, "3", True):
        with pytest.raises(ValueError):
            new_game(players=players)
    for settings in ({"thinking": "all"}, {"talk_rounds": -1}, {"talk_rounds": 1.5}, {"talk_rounds": True}):
        with pytest.raises(ValueError):
            new_game(**settings)
    for settings in ({"thinking": "round"}, {"thinking": "deal"}, {"talk_rounds": 0}, {"talk_rounds": 3}):
        new_game(**settings)  # accepted


# --- the Market -------------------------------------------------------------------------


def test_everyone_thinks_first_then_the_market_goes_clockwise_from_the_sheriffs_left():
    game = new_game(players=4)
    set_sheriff(game=game, sheriff="p3")
    phase = game.get_dialogue_phase()
    assert (phase.mode, phase.eligible) == (DialogueMode.THINKING, ("p1", "p2", "p3", "p4"))
    order = []
    while (phase := current(game=game)).phase == "set_aside":
        assert (phase.mode, phase.talk_rounds) == (DialogueMode.ACTION, 0)
        assert game.get_legal_actions(player_id="p3") == []  # the Sheriff takes no part
        order.append(phase.eligible[0])
        act(game=game, moves=[(phase.eligible[0], "done")])
    assert order == ["p4", "p1", "p2"]
    assert current(game=game).phase == "load"


def test_set_aside_offers_done_first_each_good_once_and_stops_at_five():
    hand = ["apples", "apples", "silk", "cheese", "mead", "apples"]
    game = table(hands={"p2": hand})
    assert current(game=game).eligible == ("p2",)
    assert game.get_legal_actions(player_id="p2") == [
        "done", {"set_aside": "apples"}, {"set_aside": "cheese"}, {"set_aside": "mead"}, {"set_aside": "silk"}]
    act(game=game, moves=[("p2", {"set_aside": good}) for good in ("apples", "apples", "silk", "cheese", "mead")])
    assert game.get_legal_actions(player_id="p2") == ["done"]
    assert game.state["hands"]["p2"] == ["apples"]
    assert game.get_observation(player_id="p3")["players"][1]["set_aside"] == 5  # the number is public


def test_draws_take_the_discard_piles_before_the_draw_pile_one_card_at_a_time():
    game = table(hands={"p2": ["apples", "apples", "cheese", "cheese", "pepper", "mead"]})
    game.state["discards"] = [["apples"] * 4 + ["silk"], ["cheese"] * 5]
    act(game=game, moves=[("p2", {"set_aside": "pepper"}), ("p2", {"set_aside": "mead"}), ("p2", "done")])
    assert current(game=game).phase == "draw"
    assert game.get_legal_actions(player_id="p2") == [{"draw": "deck"}, {"draw": "discard_1"}, {"draw": "discard_2"}]
    events = game.step(player_id="p2", action={"draw": "discard_1"}).public_events
    assert events == ["p2 takes Silk from discard pile 1."]
    assert game.state["discards"][0] == ["apples"] * 4
    deck_top = game.state["deck"][-1]
    events = game.step(player_id="p2", action={"draw": "deck"}).public_events
    assert events == ["p2 draws a card from the draw pile."]  # never which card
    assert sorted(game.state["hands"]["p2"]) == sorted(["apples", "apples", "cheese", "cheese", "silk", deck_top])
    assert current(game=game).phase == "discard_pile"  # back to 6
    game = table(hands={"p2": ["apples"] * 6})
    game.state["discards"][1] = []
    act(game=game, moves=[("p2", {"set_aside": "apples"})] * 3 + [("p2", "done")])
    assert game.get_legal_actions(player_id="p2") == [{"draw": "deck"}, {"draw": "discard_1"}]  # no empty pile
    act(game=game, moves=[("p2", {"draw": "deck"})])
    assert game.get_legal_actions(player_id="p2") == [{"draw": "deck"}]  # no discard pile after the draw pile


def test_set_aside_cards_go_on_the_chosen_pile_in_the_chosen_order_and_the_last_is_on_top():
    game = table(hands={"p2": ["silk", "apples", "apples", "cheese", "cheese", "cheese"]})
    act(game=game, moves=[("p2", {"set_aside": "silk"}), ("p2", {"set_aside": "apples"}),
                          ("p2", {"set_aside": "apples"}), ("p2", "done")])
    act(game=game, moves=[("p2", {"draw": "deck"})] * 3)
    assert current(game=game).phase == "discard_pile"
    assert game.get_legal_actions(player_id="p2") == [{"pile": 1}, {"pile": 2}]
    pile_before = list(game.state["discards"][1])
    act(game=game, moves=[("p2", {"pile": 2})])
    assert current(game=game).phase == "discard_order"
    assert game.get_legal_actions(player_id="p2") == [{"place": "apples"}, {"place": "silk"}]
    events = game.step(player_id="p2", action={"place": "silk"}).public_events
    assert events == ["p2 puts Silk on discard pile 2."] + ["p2 puts Apples on discard pile 2."] * 2  # no choice left
    assert game.state["discards"][1] == pile_before + ["silk", "apples", "apples"]
    observation = game.get_observation(player_id="p1")
    assert observation["discard_piles"][1] == {"top": "apples", "size": 8}
    assert current(game=game).eligible == ("p3",)  # the next merchant's Market


# --- loading and declaring --------------------------------------------------------------


def test_loading_is_simultaneous_one_card_at_a_time_and_a_full_bag_closes_itself():
    game = table(hands={"p2": ["apples"] * 5 + ["silk"], "p3": ["cheese", "cheese", "pepper", "apples", "chickens", "mead"]})
    skip_market(game=game)
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("load", DialogueMode.SIMULTANEOUS, ("p2", "p3"), 0)
    assert game.get_legal_actions(player_id="p1") == []
    assert game.get_legal_actions(player_id="p3") == [  # an empty bag needs a card: the honest one first
        {"load": "cheese"}, {"load": "apples"}, {"load": "chickens"}, {"load": "pepper"}, {"load": "mead"}]
    act(game=game, moves=[("p2", {"load": "apples"})])
    assert current(game=game).phase_id == phase.phase_id and current(game=game).eligible == ("p3",)
    act(game=game, moves=[("p3", {"load": "cheese"})])
    second = current(game=game)
    assert second.phase_id != phase.phase_id and second.eligible == ("p2", "p3")
    assert game.get_legal_actions(player_id="p3")[0] == "done"
    act(game=game, moves=[("p2", {"load": "apples"}), ("p3", "done")])
    assert current(game=game).eligible == ("p2",)
    assert game.get_observation(player_id="p2")["deciding"] == ["p2", "p3"]  # nobody learns that p3 closed
    act(game=game, moves=[("p2", {"load": "apples"})] * 3)  # the fifth card closes the bag
    assert current(game=game).phase == "declare"
    assert game.state["bags"] == {"p1": [], "p2": ["apples"] * 5, "p3": ["cheese"]}
    assert game.state["hands"]["p2"] == ["silk"]


def test_declarations_name_one_legal_type_true_type_first_and_the_game_adds_the_count():
    game = table(hands={"p2": ["apples", "apples", "silk", "cheese", "cheese", "cheese"],
                        "p3": ["pepper", "mead", "apples", "apples", "apples", "apples"]})
    skip_market(game=game)
    load_bags(game=game, bags={"p2": ["cheese", "silk", "apples", "cheese"], "p3": ["pepper", "mead"]})
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible) == ("declare", DialogueMode.ACTION, ("p2",))
    assert game.get_legal_actions(player_id="p2") == [{"declare": "cheese"}, {"declare": "apples"}, {"declare": "chickens"}]
    events = game.step(player_id="p2", action={"declare": "apples"}).public_events  # a false declaration is legal
    assert events == ["p2 declares 4 Apples."]
    assert game.state["declarations"]["p2"] == {"good": "apples", "count": 4}
    assert game.get_legal_actions(player_id="p3") == [{"declare": "apples"}, {"declare": "cheese"}, {"declare": "chickens"}]
    assert {"declare": "pepper"} not in game.get_legal_actions(player_id="p3")  # contraband is never declared
    game = table(players=4, hands={"p2": ["bread"] * 6})
    skip_market(game=game)
    load_bags(game=game, bags={"p2": ["bread"], "p3": [game.state["hands"]["p3"][0]], "p4": [game.state["hands"]["p4"][0]]})
    assert game.get_legal_actions(player_id="p2") == [
        {"declare": "bread"}, {"declare": "apples"}, {"declare": "cheese"}, {"declare": "chickens"}]


# --- inspection (this step: every bag passes) -------------------------------------------


def four_bags():
    game = table(players=4, hands={"p2": ["apples", "silk", "cheese", "cheese", "cheese", "cheese"],
                                   "p3": ["cheese"] * 6, "p4": ["chickens", "pepper", "pepper", "apples", "apples", "apples"]})
    skip_market(game=game)
    load_bags(game=game, bags={"p2": ["apples", "silk"], "p3": ["cheese"], "p4": ["chickens", "pepper", "pepper"]})
    declare_all(game=game, goods={"p2": "apples", "p3": "cheese", "p4": "chickens"})
    return game


def test_the_sheriff_chooses_the_order_and_a_passed_bag_goes_to_the_stand():
    game = four_bags()
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible) == ("pick_bag", DialogueMode.ACTION, ("p1",))
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}, {"bag": "p3"}, {"bag": "p4"}]
    act(game=game, moves=[("p1", {"bag": "p4"})])
    assert game.state["stand_legal"]["p4"] == ["chickens"]
    assert game.state["stand_contraband"]["p4"] == ["pepper", "pepper"]
    seen = game.get_observation(player_id="p2")["players"][3]
    assert (seen["stand"], seen["contraband"]) == ({"chickens": 1}, 2)  # contraband: only the number
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}, {"bag": "p3"}]
    act(game=game, moves=[("p1", {"bag": "p3"})])
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}]  # one bag left: the runner plays it


def test_after_the_last_bag_everyone_draws_back_to_six_the_sheriff_passes_left_and_everyone_thinks():
    game = four_bags()
    deck_before = len(game.state["deck"])
    act(game=game, moves=[("p1", {"bag": "p2"}), ("p1", {"bag": "p3"}), ("p1", {"bag": "p4"})])
    state = game.state
    assert all(len(state["hands"][pid]) == HAND_SIZE for pid in game.player_ids)
    assert len(state["deck"]) == deck_before - 6  # the three bags held 2, 1 and 3 cards
    assert (state["round"], state["sheriff"], state["sheriffs"]) == (2, "p2", ["p1", "p2"])
    assert state["declarations"] == {pid: None for pid in game.player_ids}
    phase = game.get_dialogue_phase()
    assert (phase.mode, phase.eligible) == (DialogueMode.THINKING, ("p1", "p2", "p3", "p4"))
    assert current(game=game).eligible == ("p3",)  # the Market starts at the new Sheriff's left


def test_an_empty_draw_pile_is_rebuilt_from_all_but_the_top_five_of_each_discard_pile():
    game = table(hands={"p2": ["apples"] * 6})
    rest = game.state["deck"]
    removed = rest[11:] + game.state["discards"][0] + game.state["discards"][1]  # out of play for this test
    game.state["discards"] = [rest[:8], rest[8:11]]
    game.state["deck"] = []
    act(game=game, moves=[("p2", {"set_aside": "apples"}), ("p2", "done")])
    assert game.get_legal_actions(player_id="p2") == [{"draw": "deck"}, {"draw": "discard_1"}, {"draw": "discard_2"}]
    top_five = rest[3:8]
    events = game.step(player_id="p2", action={"draw": "deck"}).public_events
    assert events[0].startswith("The draw pile is empty: 3 cards")
    assert game.state["discards"] == [top_five, rest[8:11]]
    assert len(game.state["deck"]) == 2 and game.state["reshuffles"] == 1
    assert all_cards(game=game) + Counter(removed) == full_deck(players=3)
    game = table(hands={"p2": ["apples"] * 6})
    game.state["discards"] = [game.state["deck"][:5], game.state["deck"][5:8]]
    game.state["deck"] = []
    act(game=game, moves=[("p2", {"set_aside": "apples"}), ("p2", "done")])
    assert game.get_legal_actions(player_id="p2") == [{"draw": "discard_1"}, {"draw": "discard_2"}]  # nothing to rebuild


# --- whole games ------------------------------------------------------------------------


def test_random_games_conserve_cards_and_gold_after_every_step():
    for players in (3, 4, 5):
        for seed in range(6):
            game = new_game(seed=seed, players=players)
            chooser = random.Random(f"{players}:{seed}")

            def check():
                assert all_cards(game=game) == full_deck(players=players)
                assert sum(game.state["gold"].values()) == START_GOLD * players
                assert all(gold >= 0 for gold in game.state["gold"].values())
                assert all(len(game.state["hands"][pid]) <= HAND_SIZE for pid in game.player_ids)
                assert all(len(game.state["bags"][pid]) <= 5 for pid in game.player_ids)

            play(game=game, choose=chooser.choice, check=check)


def test_every_game_ends_after_the_official_number_of_rounds_with_each_player_sheriff_equally():
    for players in (3, 4, 5):
        for seed in range(4):
            for choose in (random.Random(seed).choice, lambda legal: legal[0]):
                game = new_game(seed=seed, players=players)
                play(game=game, choose=choose)
                rounds = SHERIFF_TURNS[players] * players
                assert game.state["round"] == rounds + 1 and len(game.state["sheriffs"]) == rounds
                assert Counter(game.state["sheriffs"]) == {pid: SHERIFF_TURNS[players] for pid in game.player_ids}
                outcome = game.get_outcome()
                assert outcome["done"] and outcome["termination_reason"] == "all_rounds_played"
                assert sum(outcome["rewards"].values()) == pytest.approx(1.0)
                assert len(game.get_step_rewards()) == len(game.steps)


# --- scoring ----------------------------------------------------------------------------


def finished(*, players=3, legal=None, contraband=None, gold=None):
    """A game over, with these stands (lists of goods) and gold (others: empty, 50)."""
    game = new_game(players=players)
    game.state["round"] = game.state["rounds"] + 1
    for pid in game.player_ids:
        game.state["stand_legal"][pid] = list((legal or {}).get(pid, []))
        game.state["stand_contraband"][pid] = list((contraband or {}).get(pid, []))
    game.state["gold"] |= gold or {}
    return game


def bonuses(*, game):
    return {pid: row["bonus"] for pid, row in game.scores().items()}


def test_a_score_is_the_whole_stand_plus_gold_plus_bonuses_and_hands_score_nothing():
    game = finished(legal={"p1": ["apples", "cheese"]}, contraband={"p1": ["silk", "crossbow"]}, gold={"p1": 10})
    assert game.scores()["p1"] == {"goods": 2 + 3 + 8 + 9, "gold": 10, "bonus": 20 + 15, "total": 22 + 10 + 35}
    assert game.scores()["p2"] == {"goods": 0, "gold": 50, "bonus": 0, "total": 50}
    game.state["hands"]["p2"] = ["crossbow"] * 6
    assert game.scores()["p2"]["total"] == 50


def test_the_king_and_the_queen_each_get_their_bonus():
    game = finished(legal={"p1": ["apples"] * 3, "p2": ["apples"] * 2, "p3": ["apples"]})
    assert bonuses(game=game) == {"p1": 20, "p2": 10, "p3": 0}
    game = finished(legal={"p1": ["chickens"] * 2, "p2": ["chickens"]})
    assert bonuses(game=game) == {"p1": 10, "p2": 5, "p3": 0}


def test_tied_kings_share_king_plus_queen_rounded_down_and_no_queen_is_paid():
    game = finished(legal={"p1": ["apples"] * 3, "p2": ["apples"] * 3, "p3": ["apples"]})
    assert bonuses(game=game) == {"p1": 15, "p2": 15, "p3": 0}
    game = finished(legal={pid: ["cheese"] * 2 for pid in ("p1", "p2", "p3")})
    assert bonuses(game=game) == {"p1": 8, "p2": 8, "p3": 8}  # (15 + 10) // 3


def test_tied_queens_share_the_queen_rounded_down():
    game = finished(legal={"p1": ["chickens"] * 4, "p2": ["chickens"], "p3": ["chickens"]})
    assert bonuses(game=game) == {"p1": 10, "p2": 2, "p3": 2}  # 5 // 2
    game = finished(players=4, legal={"p1": ["bread"] * 2, "p2": ["bread"], "p3": ["bread"], "p4": ["bread"]})
    assert bonuses(game=game) == {"p1": 15, "p2": 3, "p3": 3, "p4": 3}  # 10 // 3


def test_no_bonus_without_the_good_and_contraband_counts_for_no_bonus():
    game = finished(legal={"p1": ["apples"]}, contraband={"p2": ["silk"] * 5})
    assert bonuses(game=game) == {"p1": 20, "p2": 0, "p3": 0}  # a lone King: nobody is Queen with none


def even_totals(*, game, totals):
    """Set gold so that each listed player's total is exactly as given."""
    for pid, total in totals.items():
        row = game.scores()[pid]
        game.state["gold"][pid] = total - row["goods"] - row["bonus"]
        assert game.state["gold"][pid] >= 0


def test_the_highest_total_wins():
    game = finished(gold={"p1": 60, "p2": 50, "p3": 40})
    outcome = game.get_outcome()
    assert outcome["winners"] == ["p1"] and outcome["rewards"] == {"p1": 1.0, "p2": 0.0, "p3": 0.0}


def test_a_tie_goes_to_the_most_legal_goods():
    game = finished(legal={"p1": ["cheese", "cheese"], "p2": ["chickens"]}, contraband={"p2": ["crossbow"]})
    even_totals(game=game, totals={"p1": 100, "p2": 100, "p3": 50})
    assert game.get_outcome()["winners"] == ["p1"]


def test_then_to_the_most_contraband():
    game = finished(legal={"p1": ["cheese"], "p2": ["cheese"]}, contraband={"p2": ["pepper"]})
    even_totals(game=game, totals={"p1": 100, "p2": 100, "p3": 50})
    assert game.get_outcome()["winners"] == ["p2"]


def test_then_the_win_is_shared():
    game = finished(legal={"p1": ["cheese"], "p2": ["apples"]}, contraband={"p1": ["mead"], "p2": ["silk"]})
    even_totals(game=game, totals={"p1": 100, "p2": 100, "p3": 50})
    outcome = game.get_outcome()
    assert outcome["winners"] == ["p1", "p2"] and outcome["rewards"] == {"p1": 0.5, "p2": 0.5, "p3": 0.0}


# --- hidden information -----------------------------------------------------------------


def seen_by(*, game, player_id):
    observation = game.get_observation(player_id=player_id)
    return json.dumps(observation), game.render_observation(observation=observation)


def other_cards(*, cards, goods):
    """As many cards as `cards`, all of one good from `goods`, and different from `cards`."""
    for good in goods:
        swapped = [good] * len(cards)
        if Counter(swapped) != Counter(cards) or not cards:
            return swapped
    raise AssertionError("unreachable")


def hide_from_viewer(*, game, owner):
    """Swap everything of `owner`'s that is hidden (same sizes), and the draw pile's order."""
    state = game.state
    state["hands"][owner] = other_cards(cards=state["hands"][owner], goods=("crossbow", "apples"))
    state["bags"][owner] = other_cards(cards=state["bags"][owner], goods=("crossbow", "apples"))
    state["stand_contraband"][owner] = other_cards(cards=state["stand_contraband"][owner], goods=("mead", "silk"))
    market = state["market"]
    if state["stage"] == "market" and market["merchants"] and market["merchants"][0] == owner:
        market["set_aside"] = other_cards(cards=market["set_aside"], goods=("crossbow", "apples"))
    state["deck"] = list(reversed(state["deck"]))


def test_no_hidden_card_appears_in_another_players_observation_or_prompt():
    """Swapping another player's hand, set-aside cards, bag and face-down contraband types (and
    the draw pile's order) changes nothing a player sees, at every step of two rounds."""
    for players in (3, 4, 5):
        for seed in range(2):
            game = new_game(seed=seed, players=players)
            chooser = random.Random(seed)
            while game.state["round"] <= 2:
                data = game.to_dict()
                for viewer in game.player_ids:
                    real = SheriffGame.from_dict(data=data)
                    assert real.get_observation(player_id=viewer)["your_hand"] == sort_goods(cards=real.state["hands"][viewer])
                    for owner in game.player_ids:
                        if owner == viewer:
                            continue
                        swapped = SheriffGame.from_dict(data=data)
                        hide_from_viewer(game=swapped, owner=owner)
                        assert seen_by(game=real, player_id=viewer) == seen_by(game=swapped, player_id=viewer)
                phase = game.get_dialogue_phase()
                player_id = phase.eligible[0]
                assert game.step(player_id=player_id, action=chooser.choice(game.get_legal_actions(player_id=player_id))).accepted


def test_the_observation_has_exactly_the_designed_fields():
    observation = new_game().get_observation(player_id="p1")
    assert set(observation) == {"you", "your_hand", "your_set_aside", "your_bag", "your_contraband", "players",
                                "round", "rounds", "sheriff", "draw_pile", "discard_piles", "unresolved", "phase",
                                "deciding", "log", "winners"}
    assert set(observation["players"][0]) == {"id", "gold", "hand_size", "set_aside", "stand", "contraband", "declared"}


def test_prompt_starts_with_the_rules_and_the_menu_is_only_in_the_answer_request():
    game = table(hands={"p2": ["apples"] * 6})
    observation = game.get_observation(player_id="p2")
    text = game.render_observation(observation=observation)
    assert text.startswith(game.get_rules(observation=observation))
    assert "Your hand: 6 Apples." in text and "Your answer" not in text
    assert "Bread" not in text  # no Bread with 3 players
    request = game.render_answer_request(legal_actions=game.get_legal_actions(player_id="p2"))
    assert '"done"' in request and '{"set_aside": "apples"}' in request
    four = new_game(players=4)
    assert "Bread 3/2" in four.get_rules(observation=four.get_observation(player_id="p1"))


# --- parsing, illegal actions and checkpoints -------------------------------------------


def test_parse_answer_reads_json_actions():
    game = new_game()
    assert game.parse_answer(player_id="p1", answer='{"set_aside": "Apples"}') == {"set_aside": "apples"}
    assert game.parse_answer(player_id="p1", answer='```json\n{"bag": "P2"}\n```') == {"bag": "p2"}
    assert game.parse_answer(player_id="p1", answer=' {"pile": 2} ') == {"pile": 2}
    assert game.parse_answer(player_id="p1", answer="Done.") == "done"
    for bad in ("", "   ", "put it all in"):
        with pytest.raises(ValueError):
            game.parse_answer(player_id="p1", answer=bad)


def test_illegal_actions_change_nothing():
    game = table(hands={"p2": ["apples"] * 6})
    before = game.to_dict()
    for player_id, action in [("p1", "done"), ("p3", "done"), ("p2", {"set_aside": "silk"}), ("p2", {"load": "apples"}),
                              ("p2", {"draw": "deck"}), ("p2", "pass"), ("p2", THINK)]:
        result = game.step(player_id=player_id, action=action)
        assert not result.accepted and result.error
    assert game.to_dict() == before


def test_save_round_trip_through_json_continues_identically():
    game = new_game(seed=3)
    chooser = random.Random(3)
    for _ in range(40):
        player_id = game.get_dialogue_phase().eligible[0]
        assert game.step(player_id=player_id, action=chooser.choice(game.get_legal_actions(player_id=player_id))).accepted
    restored = SheriffGame.from_dict(data=json.loads(json.dumps(game.to_dict())))
    assert restored.to_dict() == game.to_dict()
    for g in (game, restored):
        play(game=g, choose=lambda legal: legal[-1])
    assert restored.to_dict() == game.to_dict()


# --- the runner -------------------------------------------------------------------------


class FirstOptionModel(ChatModel):
    """Thinks "plan" and answers every action request with its first listed option."""

    def __init__(self):
        self.calls = []

    async def complete(self, *, messages):
        self.calls.append(messages)
        text = messages[-1]["content"]
        if THINK_INSTRUCTION in text:
            return reply(answer="plan")
        menu = text.split("written exactly as shown (JSON):\n")[1]
        return reply(answer=menu.splitlines()[0][2:])


def test_a_three_player_game_through_the_runner_asks_only_real_choices():
    """Every merchant keeps its hand, loads one honest card then "done", and declares truly;
    the Sheriff takes the bags clockwise. Per round: 3 thinking calls (the opening for round
    1), 2 Market, 4 loading, 2 declaring and 1 bag pick (the last bag is forced): 12 x 9."""
    game = new_game(seed=1, players=3)
    models = {pid: FirstOptionModel() for pid in game.player_ids}
    runner = Runner(game=game, models=models)
    result = asyncio.run(runner.run())
    assert result["outcome"]["done"] and result["invalid"] == {pid: 0 for pid in game.player_ids}
    assert sum(len(model.calls) for model in models.values()) == 12 * 9
    loads = [e for e in runner.record if e["kind"] == "phase" and e["phase"] == "load"]
    assert loads and all(e["mode"] == "simultaneous" and game.state["sheriffs"][0] not in e["eligible"]
                         for e in loads[:2])
    assert len(result["step_rewards"]) == len(game.steps)
