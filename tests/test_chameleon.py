import asyncio
import json
import random

import pytest

from self_play.chat_model import ChatModel, ModelCallFailed, ScriptedChatModel
from self_play.game import THINK, DialogueMode
from self_play.games import GAMES
from self_play.games.chameleon import ChameleonGame
from self_play.games.chameleon.game import REWARDS, TOPICS
from self_play.prompting import TALK_INSTRUCTION, TALK_REQUIRED_INSTRUCTION
from self_play.runner import Runner

SECRET = "owl"  # the last word of the Animals card, so a fallback guess (the first word) is never right


def new_game(*, seed=0, players=4, **settings):
    return ChameleonGame(game_config={"seed": seed, "players": players, **settings})


def rig(*, players=4, dealer="p1", chameleon="p2", topic="Animals", secret=SECRET, **settings):
    """A game with this dealer, Chameleon, topic and secret word, before the opening thinking."""
    game = new_game(players=players, **settings)
    state, ids = game.state, game.player_ids
    order = ids[ids.index(dealer):] + ids[:ids.index(dealer)]
    state.update(dealer=dealer, chameleon=chameleon, topic=topic, words=list(TOPICS[topic]), secret_word=secret,
                 clue_order=order, remaining=list(order))
    return game


def think_all(*, game):
    """Play every owed thinking turn."""
    while (phase := game.get_dialogue_phase()) is not None and phase.mode is DialogueMode.THINKING:
        for player_id in phase.eligible:
            assert game.step(player_id=player_id, action=THINK).accepted


def act(*, game, moves):
    for player_id, action in moves:
        result = game.step(player_id=player_id, action=action)
        assert result.accepted, (player_id, action, result.error)


def to_vote(*, game):
    """Think, close the clues and the debate, think again: the vote is open."""
    think_all(game=game)
    for name in ("clues", "debate"):
        phase = game.get_dialogue_phase()
        if phase.phase == name:
            act(game=game, moves=[(player_id, "done") for player_id in phase.eligible])
    think_all(game=game)
    assert game.get_dialogue_phase().phase == "vote"


def vote(*, game, votes):
    act(game=game, moves=[(voter, {"vote": target}) for voter, target in votes.items()])


def reply(*, answer):
    return f"<thinking>scratch</thinking>\n<answer>{answer}</answer>\n[STOP]"


def seen_by(*, game, player_id):
    observation = game.get_observation(player_id=player_id)
    return json.dumps(observation), game.render_observation(observation=observation)


def copy_of(*, game):
    return ChameleonGame.from_dict(data=game.to_dict())


def test_registered_by_name():
    assert GAMES["chameleon"] is ChameleonGame


# --- setup ------------------------------------------------------------------------------


def test_the_topic_cards_are_our_own_lists_of_sixteen_words():
    assert len(TOPICS) >= 8
    for topic, words in TOPICS.items():
        assert len(words) == 16 and len(set(words)) == 16, topic
        assert all(word == word.lower() and word.isalpha() for word in words), topic


def test_the_seed_picks_the_dealer_the_chameleon_the_topic_and_the_word():
    assert new_game(seed=4).state == new_game(seed=4).state
    for players in range(3, 7):
        ids = [f"p{i}" for i in range(1, players + 1)]
        games = [new_game(seed=seed, players=players) for seed in range(80)]
        assert {g.state["dealer"] for g in games} == set(ids)
        assert {g.state["chameleon"] for g in games} == set(ids)
        assert len({g.state["topic"] for g in games}) >= 5
        for g in games:
            assert g.player_ids == ids
            assert g.state["words"] == TOPICS[g.state["topic"]] and g.state["secret_word"] in g.state["words"]
            order = g.state["clue_order"]
            assert order[0] == g.state["dealer"] and sorted(order) == sorted(ids)
            assert order == ids[ids.index(order[0]):] + ids[:ids.index(order[0])]  # clockwise
    assert any(g.state["dealer"] == g.state["chameleon"] for g in games)  # the dealer may be the Chameleon


def test_three_to_six_players_and_nothing_else():
    for players in (3, 4, 5, 6):
        assert len(new_game(players=players).player_ids) == players
    for players in (0, 2, 7, 8, "4", 4.0, True):
        with pytest.raises(ValueError):
            new_game(players=players)


def test_settings_are_checked():
    for settings in ({"talk_rounds": -1}, {"talk_rounds": 1.5}, {"talk_rounds": True}):
        with pytest.raises(ValueError):
            new_game(**settings)
    for settings in ({"talk_rounds": 0}, {"talk_rounds": 1}, {"talk_rounds": 3}):
        new_game(**settings)  # accepted


# --- the phases ---------------------------------------------------------------------------


def shape(*, game):
    phase = game.get_dialogue_phase()
    return phase.phase, phase.mode, phase.eligible, phase.talk_rounds, phase.talk_required


def test_the_phases_come_in_the_documented_order():
    game = rig(dealer="p3", chameleon="p1", talk_rounds=2)
    ids = []
    everyone, dealer_order = ("p1", "p2", "p3", "p4"), ("p3", "p4", "p1", "p2")
    assert shape(game=game) == ("thinking", DialogueMode.THINKING, everyone, 0, False)
    think_all(game=game)
    assert shape(game=game) == ("clues", DialogueMode.DISCUSSION, dealer_order, 1, True)
    ids.append(game.get_dialogue_phase().phase_id)
    act(game=game, moves=[("p3", "done"), ("p4", "done")])
    assert shape(game=game) == ("clues", DialogueMode.DISCUSSION, ("p1", "p2"), 1, True)  # the same phase, two fewer
    assert game.get_dialogue_phase().phase_id == ids[0]
    act(game=game, moves=[("p1", "done"), ("p2", "done")])
    assert shape(game=game) == ("debate", DialogueMode.DISCUSSION, dealer_order, 2, False)
    ids.append(game.get_dialogue_phase().phase_id)
    act(game=game, moves=[(player_id, "done") for player_id in dealer_order])
    assert shape(game=game) == ("thinking", DialogueMode.THINKING, everyone, 0, False)
    think_all(game=game)
    assert shape(game=game) == ("vote", DialogueMode.SIMULTANEOUS, everyone, 0, False)
    ids.append(game.get_dialogue_phase().phase_id)
    vote(game=game, votes={"p1": "p2", "p2": "p1", "p3": "p1", "p4": "p2"})  # p1 (the Chameleon) and p2 are tied
    assert shape(game=game) == ("tiebreak", DialogueMode.ACTION, ("p3",), 0, False)
    ids.append(game.get_dialogue_phase().phase_id)
    act(game=game, moves=[("p3", {"choose": "p1"})])
    assert shape(game=game) == ("guess", DialogueMode.ACTION, ("p1",), 0, False)
    ids.append(game.get_dialogue_phase().phase_id)
    assert len(set(ids)) == len(ids)


def test_the_debate_is_skipped_with_zero_talk_rounds():
    game = rig(talk_rounds=0)
    think_all(game=game)
    act(game=game, moves=[(player_id, "done") for player_id in game.get_dialogue_phase().eligible])
    think_all(game=game)
    assert shape(game=game)[0] == "vote"


def test_clues_and_debate_are_closed_by_a_forced_done_and_only_the_speaker_may_act():
    game = rig()
    think_all(game=game)
    assert game.get_legal_actions(player_id="p3") == ["done"]
    assert not game.step(player_id="p3", action={"vote": "p1"}).accepted
    act(game=game, moves=[("p3", "done")])
    assert game.get_legal_actions(player_id="p3") == []
    assert not game.step(player_id="p3", action="done").accepted


# --- the rules, one scripted game each ---------------------------------------------------------


def test_the_chameleon_escapes_undetected_when_the_accused_is_innocent():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p3", "p2": "p3", "p3": "p1", "p4": "p3"})
    assert game.get_dialogue_phase() is None
    outcome = game.get_outcome()
    assert outcome["done"] and outcome["result"] == "escaped_undetected" and outcome["accused"] == "p3"
    assert outcome["rewards"] == {"p1": 0.0, "p2": 1.0, "p3": 0.0, "p4": 0.0}
    assert (outcome["chameleon"], outcome["secret_word"]) == ("p2", SECRET)
    assert game.get_legal_actions(player_id="p2") == []


def test_a_caught_chameleon_who_guesses_right_escapes():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"})
    assert shape(game=game) == ("guess", DialogueMode.ACTION, ("p2",), 0, False)
    legal = game.get_legal_actions(player_id="p2")
    assert legal == [{"guess": word} for word in TOPICS["Animals"]] and len(legal) == 16
    assert all(game.get_legal_actions(player_id=pid) == [] for pid in ("p1", "p3", "p4"))
    assert not game.step(player_id="p2", action={"guess": "table"}).accepted  # only the card's 16 words
    act(game=game, moves=[("p2", {"guess": SECRET})])
    assert game.get_dialogue_phase() is None
    outcome = game.get_outcome()
    assert outcome["result"] == "escaped_guess" and outcome["rewards"] == {"p1": 0.0, "p2": 0.5, "p3": 0.0, "p4": 0.0}


def test_a_caught_chameleon_who_guesses_wrong_loses_after_one_guess_with_four_players():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"})
    act(game=game, moves=[("p2", {"guess": "cat"})])
    assert game.get_dialogue_phase() is None
    outcome = game.get_outcome()
    assert outcome["result"] == "caught" and outcome["rewards"] == {"p1": 1.0, "p2": 0.0, "p3": 1.0, "p4": 1.0}


def test_rewards_are_the_rulebooks_optional_scores_halved():
    assert REWARDS == {"escaped_undetected": (1.0, 0.0), "escaped_guess": (0.5, 0.0), "caught": (0.0, 1.0)}
    game = rig()
    assert game.get_outcome()["rewards"] == {pid: 0.0 for pid in game.player_ids}  # zero until done
    assert not game.get_outcome()["done"]


def test_a_tie_is_broken_by_the_dealer_among_the_tied_players():
    game = rig(dealer="p1", chameleon="p3")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})  # p2 and p3 have two votes each
    assert shape(game=game) == ("tiebreak", DialogueMode.ACTION, ("p1",), 0, False)
    assert game.get_legal_actions(player_id="p1") == [{"choose": "p2"}, {"choose": "p3"}]
    assert all(game.get_legal_actions(player_id=pid) == [] for pid in ("p2", "p3", "p4"))
    assert not game.step(player_id="p1", action={"choose": "p4"}).accepted  # not tied
    act(game=game, moves=[("p1", {"choose": "p3"})])
    assert game.get_dialogue_phase().phase == "guess" and game.get_dialogue_phase().eligible == ("p3",)
    act(game=game, moves=[("p3", {"guess": "cat"})])
    assert game.get_outcome()["result"] == "caught"
    other = rig(dealer="p1", chameleon="p3")
    to_vote(game=other)
    vote(game=other, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    act(game=other, moves=[("p1", {"choose": "p2"})])  # the dealer picks the wrong one
    assert other.get_outcome()["result"] == "escaped_undetected" and other.get_dialogue_phase() is None


def test_a_dealer_who_is_tied_may_choose_itself():
    game = rig(dealer="p1", chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p3", "p2": "p1", "p3": "p1", "p4": "p3"})  # p1 and p3 tied
    assert game.get_legal_actions(player_id="p1") == [{"choose": "p1"}, {"choose": "p3"}]
    act(game=game, moves=[("p1", {"choose": "p1"})])
    assert game.get_outcome()["accused"] == "p1" and game.get_outcome()["result"] == "escaped_undetected"


def test_with_three_players_everyone_tied_lets_the_dealer_choose_among_all_three():
    game = rig(players=3, dealer="p2", chameleon="p1")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p1"})
    assert game.get_legal_actions(player_id="p2") == [{"choose": "p1"}, {"choose": "p2"}, {"choose": "p3"}]


def test_the_dealer_may_be_the_chameleon_and_breaks_ties_like_anyone():
    game = rig(dealer="p1", chameleon="p1")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    act(game=game, moves=[("p1", {"choose": "p2"})])
    assert game.get_outcome()["result"] == "escaped_undetected"
    assert game.get_outcome()["rewards"]["p1"] == 1.0


def test_with_three_players_a_caught_chameleon_gets_two_guesses_and_the_wrong_one_is_removed():
    game = rig(players=3, dealer="p1", chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p1", "p3": "p2"})
    first = game.get_dialogue_phase()
    assert first.phase == "guess" and game.get_observation(player_id="p2")["guesses_left"] == 2
    act(game=game, moves=[("p2", {"guess": "cat"})])
    second = game.get_dialogue_phase()
    assert second.phase == "guess" and second.eligible == ("p2",) and second.phase_id != first.phase_id
    legal = game.get_legal_actions(player_id="p2")
    assert len(legal) == 15 and {"guess": "cat"} not in legal and legal[0] == {"guess": "dog"}
    observation = game.get_observation(player_id="p2")
    assert (observation["guesses_left"], observation["wrong_guesses"]) == (1, ["cat"])
    assert not game.step(player_id="p2", action={"guess": "cat"}).accepted  # no repeating a wrong word
    act(game=game, moves=[("p2", {"guess": SECRET})])
    assert game.get_outcome()["result"] == "escaped_guess" and game.get_outcome()["rewards"]["p2"] == 0.5


def test_with_three_players_two_wrong_guesses_catch_the_chameleon():
    game = rig(players=3, dealer="p1", chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p1", "p3": "p2"})
    act(game=game, moves=[("p2", {"guess": "cat"})])
    assert game.get_dialogue_phase() is not None
    act(game=game, moves=[("p2", {"guess": "dog"})])
    assert game.get_dialogue_phase() is None
    assert game.get_outcome()["result"] == "caught" and game.get_outcome()["rewards"] == {"p1": 1.0, "p2": 0.0, "p3": 1.0}


def test_the_rules_text_mentions_two_guesses_only_with_three_players():
    three, four = rig(players=3, chameleon="p1"), rig(players=4, chameleon="p1")
    assert "two guesses" in three.get_rules(observation=three.get_observation(player_id="p1"))
    assert "two guesses" not in four.get_rules(observation=four.get_observation(player_id="p1"))


def test_nobody_votes_for_themselves_or_twice():
    game = rig()
    to_vote(game=game)
    for player_id in game.player_ids:
        legal = game.get_legal_actions(player_id=player_id)
        assert legal == [{"vote": pid} for pid in game.player_ids if pid != player_id]
    assert not game.step(player_id="p1", action={"vote": "p1"}).accepted
    act(game=game, moves=[("p1", {"vote": "p2"})])
    assert game.get_legal_actions(player_id="p1") == []
    assert not game.step(player_id="p1", action={"vote": "p3"}).accepted


# --- what each player sees ---------------------------------------------------------------------


def snapshots(*, game):
    """Copies of the game at the opening, in the clues, in the debate, in the vote and after two votes."""
    taken = [copy_of(game=game)]
    think_all(game=game)
    taken.append(copy_of(game=game))
    act(game=game, moves=[(player_id, "done") for player_id in game.get_dialogue_phase().eligible])
    taken.append(copy_of(game=game))
    act(game=game, moves=[(player_id, "done") for player_id in game.get_dialogue_phase().eligible])
    think_all(game=game)
    taken.append(copy_of(game=game))
    vote(game=game, votes={"p1": "p3", "p2": "p3"})
    taken.append(copy_of(game=game))
    return taken


def test_the_secret_word_never_reaches_the_chameleon_and_reaches_everyone_else():
    for chameleon in ("p1", "p2", "p4"):
        for game in snapshots(game=rig(chameleon=chameleon)):
            for viewer in game.player_ids:
                observation, text = seen_by(game=game, player_id=viewer)
                if viewer == chameleon:
                    assert json.loads(observation)["secret_word"] is None
                    assert "The secret word is" not in text and "You are the Chameleon." in text
                    assert SECRET not in " ".join(game.state["log"])
                else:
                    assert json.loads(observation)["secret_word"] == SECRET
                    assert f"The secret word is: {SECRET}." in text and "You are not the Chameleon." in text
                assert all(word in text for word in TOPICS["Animals"])  # everyone sees the topic card


def test_swapping_the_secret_word_changes_only_what_those_who_know_it_see():
    for game in snapshots(game=rig(chameleon="p2")):
        swapped = copy_of(game=game)
        swapped.state["secret_word"] = "cat"
        assert seen_by(game=game, player_id="p2") == seen_by(game=swapped, player_id="p2")  # the Chameleon
        for viewer in ("p1", "p3", "p4"):
            assert seen_by(game=game, player_id=viewer) != seen_by(game=swapped, player_id=viewer)


def test_swapping_who_the_chameleon_is_changes_nothing_for_the_other_players():
    for game in snapshots(game=rig(chameleon="p2")):
        swapped = copy_of(game=game)
        swapped.state["chameleon"] = "p4"
        for viewer in ("p1", "p3"):  # not the Chameleon either way
            assert seen_by(game=game, player_id=viewer) == seen_by(game=swapped, player_id=viewer)
        assert seen_by(game=game, player_id="p2") != seen_by(game=swapped, player_id="p2")


def test_votes_stay_hidden_until_all_are_in():
    game = rig()
    to_vote(game=game)
    act(game=game, moves=[("p1", {"vote": "p3"}), ("p2", {"vote": "p3"})])
    for viewer in game.player_ids:
        observation, text = seen_by(game=game, player_id=viewer)
        assert json.loads(observation)["votes"] is None and "voted for" not in text
        assert json.loads(observation)["deciding"] == game.player_ids  # nobody learns who has voted already
    elsewhere = rig()
    to_vote(game=elsewhere)
    act(game=elsewhere, moves=[("p1", {"vote": "p4"}), ("p2", {"vote": "p1"})])
    for viewer in game.player_ids:  # the same view whatever was voted so far
        assert seen_by(game=game, player_id=viewer) == seen_by(game=elsewhere, player_id=viewer)
    act(game=game, moves=[("p3", {"vote": "p1"}), ("p4", {"vote": "p3"})])
    observation, text = seen_by(game=game, player_id="p4")
    assert json.loads(observation)["votes"] == [["p1", "p3"], ["p2", "p3"], ["p3", "p1"], ["p4", "p3"]]
    assert "Votes: p1 voted for p3, p2 voted for p3, p3 voted for p1, p4 voted for p3." in text


def test_the_deciding_player_is_told_exactly_what_it_is_choosing():
    game = rig(chameleon="p2")
    think_all(game=game)

    def text(player_id):
        return seen_by(game=game, player_id=player_id)[1]

    assert "Your decision now: say exactly one word related to the secret word." in text("p1")
    assert "Your decision now: say exactly one word related to the topic" in text("p2")
    assert "Clue round, in this order: p1, p2, p3, p4." in text("p3")
    act(game=game, moves=[(player_id, "done") for player_id in game.player_ids])
    assert "Your decision now: say something to the table, or stay silent." in text("p3")
    act(game=game, moves=[(player_id, "done") for player_id in game.player_ids])
    think_all(game=game)
    assert "Your decision now: vote for another player. You cannot vote for yourself." in text("p3")
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    assert "Tied for the most votes: p2, p3." in text("p4") and "Your decision now" not in text("p4")
    assert "Your decision now: you are the dealer; choose which tied player is accused." in text("p1")
    act(game=game, moves=[("p1", {"choose": "p2"})])
    assert "Your decision now: guess the secret word" in text("p2") and "Guesses left: 1." in text("p2")


def test_the_answer_request_ends_with_a_tagged_example_that_parses_to_a_legal_action():
    game = rig()
    to_vote(game=game)
    for legal in (game.get_legal_actions(player_id="p1"), [{"guess": word} for word in TOPICS["Animals"]],
                  [{"choose": "p2"}, {"choose": "p3"}]):
        last_line = game.render_answer_request(legal_actions=legal).splitlines()[-1]
        assert last_line.startswith("Write your answer inside the answer tags, for example: <answer>")
        assert last_line.endswith("</answer>")
        example = last_line.split("<answer>")[1].removesuffix("</answer>")
        assert game.parse_answer(player_id="p1", answer=example) == legal[0]


def test_parse_answer_reads_json_actions():
    game = new_game()
    assert game.parse_answer(player_id="p1", answer='{"vote": "P3"}') == {"vote": "p3"}
    assert game.parse_answer(player_id="p1", answer='```json\n{"guess": "Owl"}\n```') == {"guess": "owl"}
    assert game.parse_answer(player_id="p1", answer=" Done. ") == "done"
    for bad in ("", "   ", "I vote for p3"):
        with pytest.raises(ValueError):
            game.parse_answer(player_id="p1", answer=bad)


# --- step rewards -------------------------------------------------------------------------------


def test_step_rewards_are_only_decided_once_the_game_is_over():
    game = rig()
    with pytest.raises(RuntimeError):
        game.get_step_rewards()


def test_step_rewards_when_the_chameleon_escapes_undetected():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p3", "p2": "p3", "p3": "p1", "p4": "p3"})
    final_done = [0.0, 1.0, 0.0, 0.0]  # the clues and the debate close with the speakers' final rewards
    expected = [0.0] * 4 + final_done + final_done + [0.0] * 4 + [0.0, 1.0, 0.0, 0.0]
    # the votes: p1, p3, p4 named an innocent player (0); the Chameleon's own vote earns its final reward (1)
    assert game.get_step_rewards() == expected and len(expected) == game.n_steps


def test_step_rewards_when_caught_and_the_guess_is_right():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"})
    act(game=game, moves=[("p2", {"guess": SECRET})])
    final_done = [0.0, 0.5, 0.0, 0.0]
    expected = [0.0] * 4 + final_done + final_done + [0.0] * 4 + [1.0, 0.5, 1.0, 1.0] + [1.0]
    assert game.get_step_rewards() == expected


def test_step_rewards_when_caught_and_the_guess_is_wrong():
    game = rig(chameleon="p2")
    to_vote(game=game)
    vote(game=game, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"})
    act(game=game, moves=[("p2", {"guess": "cat"})])
    final_done = [1.0, 0.0, 1.0, 1.0]
    expected = [0.0] * 4 + final_done + final_done + [0.0] * 4 + [1.0, 0.0, 1.0, 1.0] + [0.0]
    assert game.get_step_rewards() == expected


def test_step_rewards_for_a_tie_break():
    # A dealer who is not the Chameleon earns 1 for picking it, else 0.
    right = rig(dealer="p1", chameleon="p3")
    to_vote(game=right)
    vote(game=right, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    act(game=right, moves=[("p1", {"choose": "p3"}), ("p3", {"guess": "cat"})])
    rewards = right.get_step_rewards()
    assert rewards[-6:-2] == [0.0, 1.0, 0.0, 1.0]  # the votes: p3, the Chameleon, earns its final reward (0)
    assert rewards[-2:] == [1.0, 0.0]               # the tie-break names the Chameleon (1); the guess is wrong (0)
    wrong = rig(dealer="p1", chameleon="p3")
    to_vote(game=wrong)
    vote(game=wrong, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    act(game=wrong, moves=[("p1", {"choose": "p2"})])
    assert wrong.get_step_rewards()[-1] == 0.0 and wrong.get_outcome()["rewards"]["p3"] == 1.0
    # A dealer who is the Chameleon earns its final reward for the tie-break.
    own = rig(dealer="p1", chameleon="p1")
    to_vote(game=own)
    vote(game=own, votes={"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p3"})
    act(game=own, moves=[("p1", {"choose": "p2"})])
    assert own.get_step_rewards()[-1] == 1.0


def test_random_games_keep_every_invariant():
    """Uniformly random legal play, 3 to 6 players: every open phase has someone to act, the game ends,
    every step gets a reward and the final rewards are those of one of the three results."""
    for players in range(3, 7):
        for seed in range(30):
            game = new_game(seed=seed, players=players)
            chooser = random.Random(f"{players}:{seed}")
            while (phase := game.get_dialogue_phase()) is not None:
                assert phase.eligible
                player_id = phase.eligible[0]
                legal = game.get_legal_actions(player_id=player_id)
                assert legal
                assert game.step(player_id=player_id, action=chooser.choice(legal)).accepted
            outcome = game.get_outcome()
            assert outcome["done"] and outcome["result"] in REWARDS
            chameleon_reward, other_reward = REWARDS[outcome["result"]]
            assert outcome["rewards"][outcome["chameleon"]] == chameleon_reward
            assert sorted(outcome["rewards"].values()).count(other_reward) >= players - 1
            assert len(game.get_step_rewards()) == game.n_steps


# --- the runner: talk, hidden votes, rewards --------------------------------------------------------


def run_scripted(*, game, replies, max_retries=0):
    models = {pid: ScriptedChatModel(replies=replies[pid]) for pid in game.player_ids}
    runner = Runner(game=game, models=models, max_retries=max_retries)
    result = asyncio.run(runner.run())
    assert all(not model._replies for model in models.values())  # every scripted reply was used
    return result, runner, models


def player_replies(*, clue, vote_answer, debate="SILENCE", extra=()):
    """One player's calls in order: think, clue, debate, think, vote, then the extras."""
    answers = ["notes", clue, debate, "NO CHANGE", vote_answer, *extra]
    return [reply(answer=answer) for answer in answers]


def scripted_game(*, votes, extra=None, clues=("animal", "pet", "wings", "night")):
    """Dealer p1, Chameleon p2, 4 players; `votes` maps each player to its vote, `extra` adds replies."""
    extra = extra or {}
    return {pid: player_replies(clue=clue, vote_answer=json.dumps({"vote": votes[pid]}), extra=extra.get(pid, ()))
            for pid, clue in zip(("p1", "p2", "p3", "p4"), clues, strict=True)}


def calls(*, runner, player_id, kind):
    return [e for e in runner.record if e["kind"] == "call" and e["player_id"] == player_id and e["call"] == kind]


def test_a_scripted_game_clues_in_dealer_order_hidden_votes_and_step_rewards():
    votes = {"p1": "p3", "p2": "p3", "p3": "p1", "p4": "p3"}
    replies = scripted_game(votes=votes)
    replies["p2"][4:5] = [reply(answer="maybe"), reply(answer='{"vote": "p3"}')]  # an invalid vote, then a good retry
    result, runner, models = run_scripted(game=rig(), replies=replies, max_retries=1)
    assert [e["phase"] for e in runner.record if e["kind"] == "phase"] == ["thinking", "clues", "debate", "thinking", "vote"]
    clue_phase = next(e for e in runner.record if e["kind"] == "phase" and e["phase"] == "clues")
    assert (clue_phase["eligible"], clue_phase["talk_rounds"], clue_phase["talk_required"]) == (
        ["p1", "p2", "p3", "p4"], 1, True)
    assert [(e["player_id"], e["content"]) for e in runner.record if e["kind"] == "message"] == [
        ("p1", "animal"), ("p2", "pet"), ("p3", "wings"), ("p4", "night")]
    p2_clue = calls(runner=runner, player_id="p2", kind="talk")[0]["messages"][1]["content"]
    assert TALK_REQUIRED_INSTRUCTION in p2_clue and TALK_INSTRUCTION not in p2_clue and "p1 said: animal" in p2_clue
    assert "said:" not in calls(runner=runner, player_id="p1", kind="talk")[0]["messages"][1]["content"]
    retry = calls(runner=runner, player_id="p2", kind="answer")[1]["messages"][1]["content"]
    assert "Your previous answer was rejected:" in retry and "voted for" not in retry  # p1 had voted, p2 does not see how
    assert result["invalid"] == {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    assert result["outcome"]["result"] == "escaped_undetected"
    assert result["outcome"]["rewards"] == {"p1": 0.0, "p2": 1.0, "p3": 0.0, "p4": 0.0}
    final_done = [0.0, 1.0, 0.0, 0.0]
    assert result["step_rewards"] == [0.0] * 4 + final_done + final_done + [0.0] * 4 + [0.0, 1.0, 0.0, 0.0]


def test_a_scripted_game_where_the_caught_chameleon_guesses_right():
    votes = {"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"}
    result, runner, _ = run_scripted(
        game=rig(), replies=scripted_game(votes=votes, extra={"p2": [f'{{"guess": "{SECRET}"}}']}))
    assert result["outcome"]["result"] == "escaped_guess"
    assert result["outcome"]["rewards"] == {"p1": 0.0, "p2": 0.5, "p3": 0.0, "p4": 0.0}
    final_done = [0.0, 0.5, 0.0, 0.0]
    assert result["step_rewards"] == [0.0] * 4 + final_done + final_done + [0.0] * 4 + [1.0, 0.5, 1.0, 1.0] + [1.0]
    assert [e["phase"] for e in runner.record if e["kind"] == "phase"][-1] == "guess"


def test_a_scripted_game_where_the_caught_chameleon_guesses_wrong():
    votes = {"p1": "p2", "p2": "p3", "p3": "p2", "p4": "p2"}
    result, _, _ = run_scripted(game=rig(), replies=scripted_game(votes=votes, extra={"p2": ['{"guess": "cat"}']}))
    assert result["outcome"]["result"] == "caught"
    assert result["outcome"]["rewards"] == {"p1": 1.0, "p2": 0.0, "p3": 1.0, "p4": 1.0}
    final_done = [1.0, 0.0, 1.0, 1.0]
    assert result["step_rewards"] == [0.0] * 4 + final_done + final_done + [0.0] * 4 + [1.0, 0.0, 1.0, 1.0] + [0.0]


def test_a_clue_must_say_something_so_a_silent_clue_is_invalid_and_posts_nothing():
    votes = {"p1": "p3", "p2": "p3", "p3": "p1", "p4": "p3"}
    result, runner, _ = run_scripted(game=rig(), replies=scripted_game(votes=votes, clues=("animal", "pet", "SILENCE", "night")))
    assert result["invalid"] == {"p1": 0, "p2": 0, "p3": 1, "p4": 0}
    assert [e["player_id"] for e in runner.record if e["kind"] == "message"] == ["p1", "p2", "p4"]


def test_a_silent_clue_is_asked_again_and_a_retry_that_gives_one_is_not_invalid():
    votes = {"p1": "p3", "p2": "p3", "p3": "p1", "p4": "p3"}
    replies = scripted_game(votes=votes)
    replies["p3"][1:2] = [reply(answer="SILENCE"), reply(answer="wings")]
    result, runner, _ = run_scripted(game=rig(), replies=replies, max_retries=1)
    assert result["invalid"] == {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    retry = calls(runner=runner, player_id="p3", kind="talk")[1]["messages"][1]["content"]
    assert "Your previous answer was rejected:" in retry and "SILENCE is not accepted" in retry
    assert ("p3", "wings") in [(e["player_id"], e["content"]) for e in runner.record if e["kind"] == "message"]


# --- crash and resume through the runner ------------------------------------------------------------

# Dealer p1, Chameleon p3. Per player, in order. The votes tie p2 and p3 (two each): p1 breaks the tie
# by choosing p3, who guesses wrong (a 4-player game has one guess).
CRASH_REPLIES = {
    "p1": [reply(answer="p1 notes"), reply(answer="animal"), reply(answer="Who is it?"), reply(answer="NO CHANGE"),
           reply(answer='{"vote": "p2"}'), reply(answer='{"choose": "p3"}')],
    "p2": [reply(answer="p2 notes"), reply(answer="pet"), reply(answer="SILENCE"), reply(answer="NO CHANGE"),
           reply(answer='{"vote": "p3"}')],
    "p3": [reply(answer="p3 notes"), reply(answer="feathers"), reply(answer="SILENCE"), reply(answer="NO CHANGE"),
           reply(answer='{"vote": "p2"}'), reply(answer='{"guess": "cat"}')],
    "p4": [reply(answer="p4 notes"), reply(answer="night"), reply(answer="SILENCE"), reply(answer="NO CHANGE"),
           reply(answer='{"vote": "p3"}')],
}
N_CRASH_CALLS = sum(len(replies) for replies in CRASH_REPLIES.values())


class CrashSwitch:
    """Counts model calls across every seat and fails call number `fail_at` (from 0)."""

    def __init__(self, *, fail_at):
        self.fail_at = fail_at
        self.count = 0


class SeatCrashingChatModel(ChatModel):
    def __init__(self, *, replies, switch):
        self.inner = ScriptedChatModel(replies=replies)
        self.switch = switch

    async def complete(self, *, messages):
        index = self.switch.count
        self.switch.count += 1
        if index == self.switch.fail_at:
            raise ModelCallFailed("simulated crash")
        return await self.inner.complete(messages=messages)


def crash_game():
    return rig(dealer="p1", chameleon="p3")


def play_crash_game(*, models, path=None):
    runner = Runner(game=crash_game(), models=models, checkpoint_path=path)
    return runner, asyncio.run(runner.run())


def test_the_crash_reference_game_covers_clues_a_tie_break_and_a_guess():
    models = {pid: ScriptedChatModel(replies=replies) for pid, replies in CRASH_REPLIES.items()}
    runner, result = play_crash_game(models=models)
    assert all(not model._replies for model in models.values())
    assert [e["phase"] for e in runner.record if e["kind"] == "phase"] == [
        "thinking", "clues", "debate", "thinking", "vote", "tiebreak", "guess"]
    assert result["invalid"] == {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    assert result["outcome"]["result"] == "caught"
    final_done = [1.0, 1.0, 0.0, 1.0]  # p3, the Chameleon, loses
    assert result["step_rewards"] == (
        [0.0] * 4 + final_done + final_done + [0.0] * 4 + [0.0, 1.0, 0.0, 1.0] + [1.0] + [0.0])


@pytest.mark.parametrize("fail_at", range(N_CRASH_CALLS))
def test_a_chameleon_game_crashed_at_any_call_resumes_to_the_identical_game(tmp_path, fail_at):
    """Covers a crash in the opening thinking, in every clue and debate talk, in the second thinking, in the
    simultaneous vote, in the tie-break and in the guess."""
    reference, expected = play_crash_game(
        models={pid: ScriptedChatModel(replies=replies) for pid, replies in CRASH_REPLIES.items()})
    path = tmp_path / "chameleon.json"

    switch = CrashSwitch(fail_at=fail_at)
    crashing = {pid: SeatCrashingChatModel(replies=replies, switch=switch) for pid, replies in CRASH_REPLIES.items()}
    with pytest.raises(ModelCallFailed):
        play_crash_game(models=crashing, path=path)

    # Each seat continues from its first reply not yet given; journaled replies are replayed.
    remaining = {pid: ScriptedChatModel(replies=CRASH_REPLIES[pid][len(crashing[pid].inner.calls):])
                 for pid in CRASH_REPLIES}
    resumed, result = play_crash_game(models=remaining, path=path)
    assert resumed.resumed
    assert all(not model._replies for model in remaining.values())  # nothing skipped
    asked_before = sum(len(model.inner.calls) for model in crashing.values())
    asked_after = sum(len(model.calls) for model in remaining.values())
    assert asked_before + asked_after == N_CRASH_CALLS  # nothing asked twice
    assert result == expected
    assert resumed.record == json.loads(json.dumps(reference.record))
    assert resumed.game.to_dict() == reference.game.to_dict()
    assert resumed.invalid == reference.invalid
    for pid in CRASH_REPLIES:
        assert resumed.notes(player_id=pid) == reference.notes(player_id=pid)
