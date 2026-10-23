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
4. Inspection. The Sheriff chooses which bag to deal with next, one at a time. For each bag, the merchant and
   the Sheriff talk, then the merchant makes a binding offer: an amount of gold (at most what they have), then
   goods from their stand or their bag, added one at a time (never cards in hand). Promises made in talk are
   not binding. The Sheriff then either takes the offer and lets the bag pass, or refuses it and inspects the
   bag, and the offer is void. Offered goods go to the Sheriff's stand.
   - A bag that passes goes to its merchant's stand: legal goods face up, contraband face down (only their
     number is shown).
   - An inspected honest bag (every card is the declared good): the Sheriff pays the merchant the penalty of
     every good, and the goods go to the stand.
   - An inspected lying bag: the declared goods go to the stand; every other card is confiscated and put on a
     discard pile the Sheriff chooses, in the order he chooses; the merchant pays the Sheriff the penalty of
     every confiscated card.
   - A player who cannot pay a penalty in gold pays the rest with goods from their stand, one at a time:
     legal goods first, then contraband, until the debt is covered (no change is given). If the stand runs
     out, the rest is forgiven. The goods go to the receiver's stand.
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
    "negotiate": "{merchant} and the Sheriff are negotiating over {merchant}'s bag; then {merchant} offers gold.",
    "offer_goods": "{merchant} is adding goods to the offer.",
    "decide": "The Sheriff {deciding} is deciding: take the offer and let the bag pass, or inspect it.",
    "confiscate_pile": "The Sheriff {deciding} is choosing the discard pile for the confiscated goods.",
    "confiscate_order": "The Sheriff {deciding} is putting the confiscated goods on the discard pile, one at a time.",
    "pay": "{deciding} is paying the rest of a penalty with goods from their stand.",
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
    inspection = observation["inspection"]
    if inspection is not None:
        offer = inspection["offer"]
        line = f"Being dealt with: {inspection['merchant']}'s bag. Offer so far: {offer['gold']} gold"
        line += "".join(f", {describe(cards=offer[source])} from the {source}" for source in ("stand", "bag") if offer[source])
        line += "."
        if inspection["confiscated"]:
            line += f" Confiscated, still to be placed: {describe(cards=inspection['confiscated'])}."
        if inspection["debt"]:
            debt = inspection["debt"]
            line += f" {debt['payer']} still owes {debt['payee']} {debt['remaining']} gold."
        parts.append(line)
    if observation["phase"] is not None:
        merchant = inspection["merchant"] if inspection else None
        if acts_now(observation=observation):
            situation = "Your decision now: " + decision_text(observation=observation)
        else:
            situation = PHASE_TEXT[observation["phase"]].format(deciding=", ".join(observation["deciding"]), merchant=merchant)
        parts.append(f"It is round {observation['round']} of {observation['rounds']}. {situation}")
    return "\n\n".join(parts)


def acts_now(*, observation: dict[str, Any]) -> bool:
    """Whether the reader is asked for an action now. In a negotiation the Sheriff only talks:
    the merchant acts. (While bags are loaded, a merchant whose bag is closed is still listed
    as deciding, but it is never asked, so its prompt is never built.)"""
    you, phase = observation["you"], observation["phase"]
    if phase == "negotiate":
        return you == observation["inspection"]["merchant"]
    return you in observation["deciding"]


def decision_text(*, observation: dict[str, Any]) -> str:
    """Exactly what the deciding player is choosing now, in words."""
    phase, inspection = observation["phase"], observation["inspection"]
    hand, bag, set_aside = observation["your_hand"], observation["your_bag"], observation["your_set_aside"]
    gold = next(player["gold"] for player in observation["players"] if player["id"] == observation["you"])
    if phase == "set_aside":
        return (f"Set aside one more card from your hand, face down, or say done. You have set aside "
                f"{len(set_aside)} of at most 5; you will then draw back to 6 cards.")
    if phase == "draw":
        return (f"Draw one card, from the top of a discard pile or from the draw pile; you have {len(hand)} of 6 "
                f"cards. Once you draw from the draw pile, you cannot take from the discard piles.")
    if phase == "discard_pile":
        return f"Choose the discard pile your set-aside cards ({describe(cards=set_aside)}) go on, face up."
    if phase == "discard_order":
        return (f"Put one of your set-aside cards ({describe(cards=set_aside)}) on the discard pile. The last card "
                f"you put there is its new top.")
    if phase == "load":
        base = ("Add one card from your hand to your bag. Your bag is empty, so you must add one." if not bag else
                f"Add one card from your hand to your bag, or say done. Your bag holds {len(bag)} of at most 5 cards.")
        return (f"{base} Still in your hand: {describe(cards=hand)}. Already in your bag: {describe(cards=bag)}. "
                f"A card already in your bag is no longer in your hand and cannot be loaded again.")
    if phase == "declare":
        cards = f"{len(bag)} card{'s' if len(bag) != 1 else ''}"
        return (f"Declare one type of legal good for your bag of {cards} ({describe(cards=bag)}). The "
                f"declaration may be false; the game states the number of cards.")
    if phase == "pick_bag":
        return f"You are the Sheriff. Choose the next bag to deal with: {', '.join(observation['unresolved'])}."
    if phase == "negotiate":
        return (f"Offer the Sheriff any amount of gold from 0 to {gold}. Next you may add goods from your stand "
                f"or your bag to the offer.")
    if phase == "offer_goods":
        return "Add one more good from your stand or your bag to your offer, or say done. Cards in hand cannot be offered."
    merchant = inspection["merchant"]
    if phase == "decide":
        return f"You are the Sheriff. Decide only whether to take {merchant}'s offer and let the bag pass, or inspect the bag."
    confiscated = describe(cards=inspection["confiscated"])
    if phase == "confiscate_pile":
        return f"You are the Sheriff. Choose the discard pile the confiscated goods ({confiscated}) go on."
    if phase == "confiscate_order":
        return (f"You are the Sheriff. Put one of the confiscated goods ({confiscated}) on the discard pile. "
                f"The last card you put there is its new top.")
    debt = inspection["debt"]
    return (f"You owe {debt['payee']} {debt['remaining']} more gold and have no gold left. Give one good from your "
            f"stand: legal goods while you have any, then contraband. Each counts at its value, and no change is given.")


def answer_request(*, legal_actions: list[Any], lead: str | None = None) -> str:
    """The legal actions, one per line; a gold offer (every amount from 0 up) in one line. `lead`,
    if given, is a first line before them."""
    example = ("Write your answer inside the answer tags, for example: <answer>X</answer>, "
               "with X replaced by one of the actions above, copied exactly.")
    return (f"{lead}\n" if lead else "") + _menu(legal_actions=legal_actions, example=example)


ONLY_NOW = "Only the actions below are possible now."


def set_aside_lead(*, hand: list[str], set_aside: list[str]) -> str:
    return f"You still hold {describe(cards=hand)}; you have set aside {describe(cards=set_aside)} so far. {ONLY_NOW}"


def offer_goods_lead(*, gold: int, offered: list[str], from_stand: list[str], from_bag: list[str]) -> str:
    return (f"You can still add {describe(cards=from_stand)} from your stand and {describe(cards=from_bag)} from your "
            f"bag; your offer so far is {gold} gold and {describe(cards=offered) if offered else 'no goods'}. {ONLY_NOW}")


def decide_lead(*, gold: int, goods: list[str]) -> str:
    return (f"The offer on the table: {gold} gold, {describe(cards=goods) if goods else 'no goods'}. "
            f"What players said is not binding; only this offer counts.")


def _menu(*, legal_actions: list[Any], example: str) -> str:
    amounts = [action.get("gold") for action in legal_actions if isinstance(action, dict) and set(action) == {"gold"}]
    if amounts and len(amounts) == len(legal_actions) and amounts == list(range(len(amounts))):
        return (f'Your answer is your gold offer, written exactly as shown (JSON):\n- {{"gold": n}} with n a whole '
                f"number from 0 to {amounts[-1]}\n{example}")
    options = "\n".join(f"- {json.dumps(action)}" for action in legal_actions)
    return f"Your answer is one of these actions, written exactly as shown (JSON):\n{options}\n{example}"


def _normal(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key).lower(): _normal(item) for key, item in value.items()}
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
    return _normal(action)
