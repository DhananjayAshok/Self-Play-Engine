"""What a Sheriff of Nottingham player reads, and how its answers are read. Built only from an
observation, so it can show nothing the observation hides."""

import json
from collections import Counter
from typing import Any

NAMES = {"apples": "Apples", "cheese": "Cheese", "bread": "Bread", "chickens": "Chickens",
         "pepper": "Pepper", "mead": "Mead", "silk": "Silk", "crossbow": "Crossbow"}
"""Display names, in the game's fixed order of goods."""

RULES = """You are playing Sheriff of Nottingham with {players} players, p1 to p{players} in seat order. Each player is the
Sheriff {sheriff_turns} times, so the game lasts {rounds} rounds.

Goods (value / penalty, in gold). Legal goods: {legal}. Contraband: Pepper 6/4, Mead 7/4, Silk 8/4,
Crossbow 9/4. Each player starts with 50 gold and 6 cards in hand. Gold is public; cards in hand are seen
only by their owner. There is a face-down draw pile and two face-up discard piles; only the top card of
a discard pile can be taken.

Each round has one Sheriff; everyone else is a merchant. A round has four parts:
1. Market. Each merchant in turn, clockwise from the Sheriff's left, may set aside up to 5 cards from their
   hand, face down, then draws back to 6 cards one at a time, from the top of either discard pile or from
   the draw pile. Every card wanted from the discard piles must be taken before any card from the draw pile.
   Then the merchant puts the set-aside cards face up on one discard pile of their choice, in the order they
   choose; the last card placed is the new top.
2. Loading. All merchants at the same time secretly put 1 to 5 cards from their hand into their bag, one
   card at a time.
3. Declaration. Each merchant in turn, clockwise from the Sheriff's left, declares the exact number of cards
   in their bag (the game states it) and one type of legal good. The declaration may be false, but
   contraband can never be declared.
4. Inspection. The Sheriff chooses which bag to deal with next, one at a time. A bag that passes goes to its
   merchant's stand: legal goods face up, contraband face down (only their number is shown).
At the end of the round everyone draws back to 6 cards from the draw pile, and the next player clockwise
becomes the Sheriff. When the draw pile is empty, each discard pile except its top five cards is shuffled
into a new draw pile.

Scoring: the value of every good in your stand, plus your gold, plus King and Queen bonuses. For each legal
good, the player with the most of it in their stand is King and the next most is Queen: {bonuses}.
Players tied for King share King plus Queen, rounded down, and no Queen is paid; players tied for Queen
share the Queen bonus, rounded down. Cards in hand score nothing. The highest score wins; ties go to the
most legal goods, then the most contraband, then the win is shared. Talk is allowed but never binding."""

PHASE_TEXT = {
    "set_aside": "{deciding} is choosing cards to set aside in the Market.",
    "draw": "{deciding} is drawing back to 6 cards in the Market.",
    "discard_pile": "{deciding} is choosing the discard pile for their set-aside cards.",
    "discard_order": "{deciding} is putting their set-aside cards on the discard pile, one at a time.",
    "load": "The merchants ({deciding}) are loading their bags, one card at a time.",
    "declare": "{deciding} is declaring their bag.",
    "pick_bag": "The Sheriff {deciding} is choosing the next bag to deal with.",
}


def rules_text(*, players: int, sheriff_turns: int, rounds: int, bread: bool) -> str:
    legal = "Apples 2/2, Cheese 3/2, " + ("Bread 3/2, " if bread else "") + "Chickens 4/2"
    bonuses = "Apples 20/10, Cheese 15/10, " + ("Bread 15/10, " if bread else "") + "Chickens 10/5"
    return RULES.format(players=players, sheriff_turns=sheriff_turns, rounds=rounds, legal=legal, bonuses=bonuses)


def describe(*, cards: list[str]) -> str:
    """'2 Apples, 1 Silk', in the game's order of goods; 'nothing' for no cards."""
    counts = Counter(cards)
    parts = [f"{counts[good]} {name}" for good, name in NAMES.items() if counts[good]]
    return ", ".join(parts) if parts else "nothing"


def _player_line(*, player: dict[str, Any], you: str, sheriff: str) -> str:
    tags = [tag for tag, on in (("you", player["id"] == you), ("Sheriff", player["id"] == sheriff)) if on]
    name = f"{player['id']} ({', '.join(tags)})" if tags else player["id"]
    stand = describe(cards=[good for good, count in player["stand"].items() for _ in range(count)])
    line = (f"- {name}: {player['gold']} gold, {player['hand_size']} cards in hand; stand: {stand} face up, "
            f"{player['contraband']} contraband face down")
    if player["set_aside"]:
        line += f"; {player['set_aside']} cards set aside"
    if player["declared"]:
        line += f"; declared {player['declared']['count']} {NAMES[player['declared']['good']]}"
    return line


def render_prompt(*, rules: str, observation: dict[str, Any]) -> str:
    you, sheriff = observation["you"], observation["sheriff"]
    role = "the Sheriff" if you == sheriff else "a merchant"
    mine = [f"You are {you}, {role} this round. Your hand: {describe(cards=observation['your_hand'])}."]
    if observation["your_set_aside"]:
        mine.append(f"Your set-aside cards (face down): {describe(cards=observation['your_set_aside'])}.")
    if observation["your_bag"]:
        mine.append(f"Your bag: {describe(cards=observation['your_bag'])}.")
    mine.append(f"Your face-down contraband: {describe(cards=observation['your_contraband'])}.")
    piles = []
    for number, pile in enumerate(observation["discard_piles"], start=1):
        top = f"top card {NAMES[pile['top']]}, {pile['size']} cards" if pile["top"] else "empty"
        piles.append(f"Discard pile {number}: {top}.")
    table = "\n".join(_player_line(player=player, you=you, sheriff=sheriff) for player in observation["players"])
    parts = [
        rules,
        " ".join(mine),
        "Players:\n" + table,
        f"Draw pile: {observation['draw_pile']} cards. " + " ".join(piles),
        "What has happened so far:\n" + "\n".join(f"- {line}" for line in observation["log"]),
    ]
    if observation["unresolved"]:
        parts.append("Bags still to be dealt with: " + ", ".join(observation["unresolved"]) + ".")
    if observation["phase"] is not None:
        situation = PHASE_TEXT[observation["phase"]].format(deciding=", ".join(observation["deciding"]))
        parts.append(f"It is round {observation['round']} of {observation['rounds']}. {situation}")
    return "\n\n".join(parts)


def answer_request(*, legal_actions: list[Any]) -> str:
    options = "\n".join(f"- {json.dumps(action)}" for action in legal_actions)
    return f"Your answer is one of these actions, written exactly as shown (JSON):\n{options}"


def _normal(value: Any) -> Any:
    return value.lower() if isinstance(value, str) else value


def parse_action(*, answer: str) -> Any:
    """The action as JSON, with a surrounding code fence ignored and strings in lower case. A
    bare word becomes a string action (e.g. "done"). Reading is not checking legality: `step`
    does that."""
    text = answer.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        raise ValueError("the answer is empty; write one of the listed actions exactly")
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        word = text.strip(".!\"'`*").strip().lower()
        if word.isidentifier():
            return word
        raise ValueError("write one of the listed actions exactly, as JSON") from None
    if isinstance(action, dict):
        return {str(key).lower(): _normal(value) for key, value in action.items()}
    return _normal(action)
