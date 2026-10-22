"""What a Coup player reads, and how its answers are read. Built only from an observation,
so it can show nothing the observation hides."""

import json
from typing import Any

RULES = """You are playing Coup with {players} players. The game ends after {max_turns} turns at most.
Each player starts with 2 face-down character cards (their influence) and 2 coins; with 2 players the
starting player gets only 1 coin. The deck holds 15 cards: 3 each of Duke, Assassin, Captain, Ambassador
and Contessa. Coins are public; face-down cards are seen only by their owner.

Players take turns in seat order. On your turn you take exactly one action. You may claim any character,
whether or not you hold it:
- Income: take 1 coin.
- Foreign Aid: take 2 coins. Any other player may block it by claiming Duke.
- Coup: pay 7 coins; the target loses one influence. Cannot be blocked or challenged. If you start your
  turn with 10 or more coins, you must launch a Coup.
- Tax (claims Duke): take 3 coins.
- Assassinate (claims Assassin): pay 3 coins; the target loses one influence. The target may block by
  claiming Contessa.
- Steal (claims Captain): take 2 coins from the target (1 if it has 1, 0 if it has none). The target may
  block by claiming Captain or Ambassador.
- Exchange (claims Ambassador): draw 2 cards from the deck, keep as many cards as you had, return the rest.

Challenges: when someone claims a character, for an action or a block, any other player may challenge.
The claimant then shows the card if they hold it: they shuffle it into the deck and draw a new card, and
the challenger loses one influence. If they cannot show it, the claimant loses one influence and the
action or block fails. A challenged action that fails refunds its cost; a blocked action does not.
Challenges come before blocks: an action that survives its challenge can still be blocked.

A player who loses an influence chooses which of their face-down cards to reveal; revealed cards stay face up.
A player with no face-down cards is out. The last player with influence wins; if the turn limit is
reached first, nobody wins. Talk is allowed but never binding."""

ACTION_TEXT = {"income": "Income", "foreign_aid": "Foreign Aid", "coup": "Coup", "tax": "Tax (claims Duke)",
               "assassinate": "Assassinate (claims Assassin)", "steal": "Steal (claims Captain)",
               "exchange": "Exchange (claims Ambassador)"}

CLAIM_CARD = {"tax": "Duke", "assassinate": "Assassin", "steal": "Captain", "exchange": "Ambassador"}
BLOCK_CARDS = {"foreign_aid": "Duke", "steal": "Captain or Ambassador", "assassinate": "Contessa"}
NOT_YOUR_TURN = "You are not taking a turn, so you cannot declare an action now."

PHASE_TEXT = {
    "action": "{actor} is choosing an action.",
    "challenge_action": "Waiting for {deciding} to pass or challenge {actor}'s claim.",
    "block_action": "Waiting for {deciding} to pass or block.",
    "challenge_block": "Waiting for {deciding} to pass or challenge the block.",
    "lose_influence": "{deciding} must reveal one face-down card.",
    "exchange": "{actor} is choosing which cards to keep.",
}


def rules_text(*, players: int, max_turns: int) -> str:
    return RULES.format(players=players, max_turns=max_turns)


def _card_list(cards: list[str]) -> str:
    return ", ".join(card.capitalize() for card in cards) if cards else "none"


def render_prompt(*, rules: str, observation: dict[str, Any]) -> str:
    you = observation["you"]
    table = []
    for player in observation["players"]:
        status = "" if player["alive"] else " (out)"
        name = f"{player['id']} (you)" if player["id"] == you else player["id"]
        table.append(
            f"- {name}{status}: {player['coins']} coins, {player['hidden_cards']} face-down cards, "
            f"revealed: {_card_list(player['revealed'])}"
        )
    parts = [
        rules,
        f"You are {you}. Your face-down cards: {_card_list(observation['your_cards'])}.",
        "Players:\n" + "\n".join(table) + f"\nCards left in the deck: {observation['deck_size']}.",
        "What has happened so far:\n" + "\n".join(f"- {line}" for line in observation["log"]),
    ]
    if observation["your_drawn_cards"]:
        parts.append(f"You drew: {_card_list(observation['your_drawn_cards'])}.")
    pending = observation["pending"]
    if pending is not None:
        action = pending["action"]
        target = f" targeting {action['target']}" if "target" in action else ""
        line = f"Being resolved: {pending['actor']} declared {ACTION_TEXT[action['action']]}{target}."
        if pending["block"]:
            line += f" {pending['block']['by']} blocks, claiming {pending['block']['card'].capitalize()}."
        parts.append(line)
    if observation["phase"] is not None:
        if you in observation["deciding"]:
            situation = "Your decision now: " + decision_text(observation=observation)
        else:
            situation = PHASE_TEXT[observation["phase"]].format(
                actor=observation["actor"], deciding=", ".join(observation["deciding"]))
        parts.append(f"It is turn {observation['turn']} of at most {observation['max_turns']}. {situation}")
    return "\n\n".join(parts)


def decision_text(*, observation: dict[str, Any]) -> str:
    """Exactly what the deciding player is choosing now, in words."""
    phase, you, pending = observation["phase"], observation["you"], observation["pending"]
    if phase == "action":
        opponents = [p["id"] for p in observation["players"] if p["alive"] and p["id"] != you]
        return f"it is your turn; choose one action. Opponents still in the game: {', '.join(opponents)}."
    if phase == "lose_influence":
        return "you lose an influence; choose which of your face-down cards to reveal."
    if phase == "exchange":
        return "choose which cards to keep from your face-down cards and the two you drew; the rest go back to the deck."
    action, actor = pending["action"], pending["actor"]
    declared = f"{actor} declared {ACTION_TEXT[action['action']].split(' (')[0]}"
    if "target" in action:
        declared += f" targeting {'you' if action['target'] == you else action['target']}"
    if phase == "challenge_action":
        card = CLAIM_CARD[action["action"]]
        return f"{NOT_YOUR_TURN} Decide only whether to challenge {actor}'s claim to hold {card} ({declared})."
    if phase == "block_action":
        cards = BLOCK_CARDS[action["action"]]
        return f"{NOT_YOUR_TURN} Decide only whether to block it ({declared}) by claiming {cards}, or to pass."
    blocker, card = pending["block"]["by"], pending["block"]["card"].capitalize()
    return (f"{NOT_YOUR_TURN} Decide only whether to challenge {blocker}'s claim to hold {card}, "
            f"made to block this: {declared}.")


def answer_request(*, legal_actions: list[Any]) -> str:
    options = "\n".join(f"- {json.dumps(action)}" for action in legal_actions)
    return (f"Your answer is one of these actions, written exactly as shown (JSON):\n{options}\n"
            f"Write your answer inside the answer tags, for example: <answer>X</answer>, "
            f"with X replaced by one of the actions above, copied exactly.")


def _normal(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return sorted(_normal(item) for item in value) if all(isinstance(item, str) for item in value) else value
    return value


def parse_action(*, answer: str) -> Any:
    """The action as JSON, with a surrounding code fence ignored, in lower case, with card
    lists sorted. A bare word becomes a string action (e.g. "pass"). Reading is not checking
    legality: `step` does that."""
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
