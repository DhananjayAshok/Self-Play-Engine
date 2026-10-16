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
from self_play.prompting import TALK_INSTRUCTION, THINK_INSTRUCTION
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
    if state["inspection"]:
        cards += state["inspection"]["confiscated"]
    for pid in game.player_ids:
        cards += state["hands"][pid] + state["bags"][pid] + state["stand_legal"][pid] + state["stand_contraband"][pid]
    return Counter(cards)


def set_sheriff(*, game, sheriff):
    """Make `sheriff` this round's Sheriff, before anyone has acted in the Market."""
    game.state["sheriff"] = sheriff
    game.state["sheriffs"] = [sheriff]
    game.state["market"] = game._new_market(merchants=game.merchants())
    game.state["log"] = [f"The game begins. Round 1 of {game.state['rounds']}: {sheriff} is the Sheriff."]


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
    game = new_game(players=players, **{"talk_rounds": 0, **settings})
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
    """Play to the end; `choose(legal)` picks each action, `check()` runs after every step. Like
    the runner, a player is asked only while the phase it was eligible in is still open."""
    while (phase := game.get_dialogue_phase()) is not None:
        assert phase.eligible
        for player_id in phase.eligible:
            now = game.get_dialogue_phase()
            if now is None or now.phase_id != phase.phase_id or player_id not in now.eligible:
                continue  # e.g. the Sheriff in a negotiation: the merchant's offer closed it
            legal = game.get_legal_actions(player_id=player_id)
            assert legal and len(legal) == len({json.dumps(option) for option in legal})
            assert game.step(player_id=player_id, action=choose(legal)).accepted
            if check:
                check()


def offer(*, game, merchant, gold=0, goods=()):
    """The Sheriff takes up `merchant`'s bag, and the merchant offers this gold and these
    goods ((source, good) pairs); the game then waits for the Sheriff's decision."""
    sheriff = game.state["sheriff"]
    act(game=game, moves=[(sheriff, {"bag": merchant})])
    think_all(game=game)
    act(game=game, moves=[(merchant, {"gold": gold})] + [(merchant, {"add": {"from": source, "good": good}})
                                                          for source, good in goods] + [(merchant, "done")])


def pass_bag(*, game, merchant, gold=0, goods=()):
    offer(game=game, merchant=merchant, gold=gold, goods=goods)
    act(game=game, moves=[(game.state["sheriff"], "pass")])


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


# --- inspection -------------------------------------------------------------------------


def four_bags(**settings):
    """p1 is Sheriff. p2 lies (Apples and Silk, declared Apples), p3 is honest (one Cheese),
    p4 lies (Chickens and two Pepper, declared Chickens)."""
    game = table(players=4, hands={"p2": ["apples", "silk", "cheese", "cheese", "cheese", "cheese"],
                                   "p3": ["cheese"] * 6, "p4": ["chickens", "pepper", "pepper", "apples", "apples", "apples"]},
                 **settings)
    skip_market(game=game)
    load_bags(game=game, bags={"p2": ["apples", "silk"], "p3": ["cheese"], "p4": ["chickens", "pepper", "pepper"]})
    declare_all(game=game, goods={"p2": "apples", "p3": "cheese", "p4": "chickens"})
    return game


def from_deck(*, game, cards):
    """Take these cards out of the draw pile, so a test can put them in a stand."""
    for card in cards:
        game.state["deck"].remove(card)
    return list(cards)


def test_the_sheriff_chooses_the_order_and_a_passed_bag_goes_to_the_stand():
    game = four_bags()
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible) == ("pick_bag", DialogueMode.ACTION, ("p1",))
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}, {"bag": "p3"}, {"bag": "p4"}]
    pass_bag(game=game, merchant="p4")
    assert game.state["stand_legal"]["p4"] == ["chickens"]
    assert game.state["stand_contraband"]["p4"] == ["pepper", "pepper"]
    seen = game.get_observation(player_id="p2")["players"][3]
    assert (seen["stand"], seen["contraband"]) == ({"chickens": 1}, 2)  # contraband: only the number
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}, {"bag": "p3"}]
    pass_bag(game=game, merchant="p3")
    assert game.get_legal_actions(player_id="p1") == [{"bag": "p2"}]  # one bag left: the runner plays it


def test_merchant_and_sheriff_talk_then_the_merchant_offers_gold_then_goods_then_the_sheriff_decides():
    game = four_bags(talk_rounds=2)
    act(game=game, moves=[("p1", {"bag": "p2"})])
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("negotiate", DialogueMode.DISCUSSION, ("p2", "p1"), 2)
    legal = game.get_legal_actions(player_id="p2")
    assert legal == [{"gold": n} for n in range(51)]  # never more than the merchant has
    assert game.get_legal_actions(player_id="p1") == []  # the Sheriff talks here but does not act
    request = game.render_answer_request(legal_actions=legal)
    assert '{"gold": n}' in request and "from 0 to 50" in request and '{"gold": 7}' not in request
    assert not game.step(player_id="p2", action={"gold": 51}).accepted
    act(game=game, moves=[("p2", {"gold": 5})])
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("offer_goods", DialogueMode.ACTION, ("p2",), 0)
    assert game.get_legal_actions(player_id="p2") == [  # its stand is empty; never cards in hand
        "done", {"add": {"from": "bag", "good": "apples"}}, {"add": {"from": "bag", "good": "silk"}}]
    act(game=game, moves=[("p2", {"add": {"from": "bag", "good": "silk"}})])
    assert game.get_legal_actions(player_id="p2") == ["done", {"add": {"from": "bag", "good": "apples"}}]
    act(game=game, moves=[("p2", "done")])
    phase = current(game=game)
    assert (phase.phase, phase.mode, phase.eligible, phase.talk_rounds) == ("decide", DialogueMode.ACTION, ("p1",), 0)
    assert game.get_legal_actions(player_id="p1") == ["pass", "inspect"]
    assert game.get_observation(player_id="p3")["inspection"]["offer"] == {"gold": 5, "stand": [], "bag": ["silk"]}


def test_taking_an_offer_moves_its_gold_and_goods_to_the_sheriff_and_the_rest_of_the_bag_passes():
    game = four_bags()
    game.state["stand_legal"]["p2"] = from_deck(game=game, cards=["cheese"])
    game.state["stand_contraband"]["p2"] = from_deck(game=game, cards=["mead"])
    pass_bag(game=game, merchant="p2", gold=7,
             goods=[("stand", "cheese"), ("stand", "mead"), ("bag", "silk")])
    state = game.state
    assert (state["gold"]["p2"], state["gold"]["p1"]) == (43, 57)
    assert (state["stand_legal"]["p1"], state["stand_contraband"]["p1"]) == (["cheese"], ["mead", "silk"])
    assert (state["stand_legal"]["p2"], state["stand_contraband"]["p2"], state["bags"]["p2"]) == (["apples"], [], [])
    assert all_cards(game=game) == full_deck(players=4)


def test_an_inspected_honest_bag_costs_the_sheriff_its_penalties_and_the_offer_is_void():
    game = four_bags()
    offer(game=game, merchant="p3", gold=4)
    events = game.step(player_id="p1", action="inspect").public_events
    assert events[0] == "p1 refuses the offer and inspects p3's bag: 1 Cheese."
    assert game.state["stand_legal"]["p3"] == ["cheese"]
    assert (game.state["gold"]["p1"], game.state["gold"]["p3"]) == (48, 52)  # the penalty, not the offer
    assert game.state["inspection"] is None and current(game=game).phase == "pick_bag"


def test_an_inspected_lying_bag_keeps_the_declared_goods_and_the_sheriff_places_the_rest():
    game = four_bags()
    offer(game=game, merchant="p4", gold=3)
    act(game=game, moves=[("p1", "inspect")])
    assert game.state["stand_legal"]["p4"] == ["chickens"]  # the declared goods
    phase = current(game=game)
    assert (phase.phase, phase.eligible) == ("confiscate_pile", ("p1",))
    assert game.get_legal_actions(player_id="p1") == [{"pile": 1}, {"pile": 2}]
    act(game=game, moves=[("p1", {"pile": 2})])  # one good: placed without asking
    assert game.state["discards"][1][-2:] == ["pepper", "pepper"]
    assert (game.state["gold"]["p4"], game.state["gold"]["p1"]) == (42, 58)  # 2 x 4 in penalties
    assert all_cards(game=game) == full_deck(players=4)


def lying_bag(*, bag, declared, gold=50, legal=(), contraband=()):
    """3 players, p1 Sheriff: p2 loads `bag` and declares `declared`, with this gold and stand."""
    game = table(hands={"p2": list(bag) + ["apples"] * (6 - len(bag))})
    skip_market(game=game)
    load_bags(game=game, bags={"p2": list(bag), "p3": [game.state["hands"]["p3"][0]]})
    act(game=game, moves=[("p2", {"declare": declared})])
    act(game=game, moves=[("p3", game.get_legal_actions(player_id="p3")[0])])
    game.state["gold"]["p2"] = gold
    game.state["stand_legal"]["p2"] = from_deck(game=game, cards=legal)
    game.state["stand_contraband"]["p2"] = from_deck(game=game, cards=contraband)
    offer(game=game, merchant="p2")
    act(game=game, moves=[("p1", "inspect")])
    if len(set(bag) - {declared}) == 1:
        act(game=game, moves=[("p1", {"pile": 1})])  # one good confiscated: it goes on pile 1 by itself
    return game


def test_the_sheriff_chooses_the_order_of_confiscated_goods_and_the_last_is_on_top():
    game = lying_bag(bag=["cheese", "silk", "mead"], declared="apples")
    act(game=game, moves=[("p1", {"pile": 1})])
    assert game.get_legal_actions(player_id="p1") == [{"place": "cheese"}, {"place": "mead"}, {"place": "silk"}]
    act(game=game, moves=[("p1", {"place": "mead"}), ("p1", {"place": "silk"})])  # Cheese, the last, goes by itself
    assert game.state["discards"][0][-3:] == ["mead", "silk", "cheese"]
    assert game.state["gold"]["p2"] == 50 - (2 + 4 + 4)


def test_a_player_short_of_gold_pays_with_legal_goods_first_cheapest_first_and_gets_no_change():
    game = lying_bag(bag=["silk"], declared="apples", gold=1, legal=["chickens", "apples", "apples"],
                     contraband=["crossbow"])
    phase = current(game=game)
    assert (phase.phase, phase.eligible) == ("pay", ("p2",))  # 1 gold paid, 3 still owed
    assert game.state["inspection"]["debt"] == {"payer": "p2", "payee": "p1", "remaining": 3}
    assert game.get_legal_actions(player_id="p2") == [{"pay": "apples"}, {"pay": "chickens"}]  # no contraband yet
    act(game=game, moves=[("p2", {"pay": "chickens"})])  # worth 4: covers 3, no change
    assert game.state["stand_legal"]["p1"] == ["chickens"]
    assert game.state["stand_legal"]["p2"] == ["apples", "apples"] and game.state["stand_contraband"]["p2"] == ["crossbow"]
    assert (game.state["gold"]["p2"], game.state["gold"]["p1"]) == (0, 51)
    assert current(game=game).phase == "pick_bag"


def test_contraband_pays_only_once_no_legal_goods_are_left():
    game = lying_bag(bag=["silk"], declared="apples", gold=0, legal=["apples"], contraband=["crossbow", "pepper"])
    act(game=game, moves=[("p2", {"pay": "apples"})])  # 2 of the 4
    assert game.get_legal_actions(player_id="p2") == [{"pay": "pepper"}, {"pay": "crossbow"}]
    act(game=game, moves=[("p2", {"pay": "pepper"})])
    assert game.state["stand_contraband"]["p1"] == ["pepper"] and game.state["stand_contraband"]["p2"] == ["crossbow"]


def test_the_rest_of_a_debt_is_forgiven_once_the_stand_is_empty():
    game = lying_bag(bag=["silk"], declared="apples", gold=0, legal=["apples"])
    events = game.step(player_id="p2", action={"pay": "apples"}).public_events
    assert "p2's stand is empty: the remaining 2 gold is forgiven." in events
    assert game.state["inspection"] is None and game.state["gold"]["p2"] == 0


def test_a_sheriff_short_of_gold_pays_an_honest_merchant_with_goods_too():
    game = table(hands={"p2": ["cheese"] * 6})
    skip_market(game=game)
    load_bags(game=game, bags={"p2": ["cheese"] * 3, "p3": [game.state["hands"]["p3"][0]]})
    act(game=game, moves=[("p2", {"declare": "cheese"})])
    act(game=game, moves=[("p3", game.get_legal_actions(player_id="p3")[0])])
    game.state["gold"]["p1"] = 1
    game.state["stand_legal"]["p1"] = from_deck(game=game, cards=["chickens", "chickens"])
    offer(game=game, merchant="p2")
    act(game=game, moves=[("p1", "inspect")])  # owes 3 x 2 = 6: pays 1 gold, then goods
    assert (current(game=game).phase, current(game=game).eligible) == ("pay", ("p1",))
    act(game=game, moves=[("p1", {"pay": "chickens"}), ("p1", {"pay": "chickens"})])  # 4 + 4 covers 5
    assert game.state["stand_legal"]["p2"] == ["cheese"] * 3 + ["chickens"] * 2


def test_thinking_deal_asks_the_merchant_and_the_sheriff_before_each_negotiation():
    for thinking, expected in [("deal", ("p1", "p3")), ("round", None)]:
        game = four_bags(thinking=thinking)
        act(game=game, moves=[("p1", {"bag": "p3"})])
        phase = game.get_dialogue_phase()
        if expected:
            assert (phase.mode, phase.eligible) == (DialogueMode.THINKING, expected)
        else:
            assert phase.phase == "negotiate"


def test_after_the_last_bag_everyone_draws_back_to_six_the_sheriff_passes_left_and_everyone_thinks():
    game = four_bags()
    deck_before = len(game.state["deck"])
    for merchant in ("p2", "p3", "p4"):
        pass_bag(game=game, merchant=merchant)
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
                                "round", "rounds", "sheriff", "draw_pile", "discard_piles", "unresolved", "inspection", "phase",
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
        if TALK_INSTRUCTION in text:
            return reply(answer="A fair bag.")
        first = text.split("written exactly as shown (JSON):\n")[1].splitlines()[0][2:]
        return reply(answer='{"gold": 0}' if first.startswith('{"gold": n}') else first)


def test_a_three_player_game_through_the_runner_asks_only_real_choices():
    """Every merchant keeps its hand, loads one honest card then "done", declares truly,
    offers nothing, and the Sheriff takes the bags clockwise and lets them pass. Per round:
    3 thinking calls (the opening for round 1), 2 Market, 4 loading, 2 declaring, 1 bag pick
    (the last bag is forced), and per bag 4 talk calls (2 rounds x merchant and Sheriff), the
    gold offer, "done" and the decision: 12 + 2 x 7 = 26 calls, 9 rounds."""
    game = new_game(seed=1, players=3)
    models = {pid: FirstOptionModel() for pid in game.player_ids}
    runner = Runner(game=game, models=models)
    result = asyncio.run(runner.run())
    assert result["outcome"]["done"] and result["invalid"] == {pid: 0 for pid in game.player_ids}
    assert sum(len(model.calls) for model in models.values()) == 26 * 9
    loads = [e for e in runner.record if e["kind"] == "phase" and e["phase"] == "load"]
    assert loads and all(e["mode"] == "simultaneous" and game.state["sheriffs"][0] not in e["eligible"]
                         for e in loads[:2])
    negotiations = {e["phase_id"]: e["eligible"] for e in runner.record if e["kind"] == "phase" and e["phase"] == "negotiate"}
    assert len(negotiations) == 2 * 9
    for phase_id, (merchant, sheriff) in negotiations.items():
        calls = [(e["player_id"], e["call"]) for e in runner.record if e["kind"] == "call" and e["phase_id"] == phase_id]
        assert calls == [(merchant, "talk"), (sheriff, "talk")] * 2 + [(merchant, "answer")]  # the Sheriff never acts here
    assert len(result["step_rewards"]) == len(game.steps)
