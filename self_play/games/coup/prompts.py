"""What a Coup player reads, and how its answers are read. Built only from an observation,
so it can show nothing the observation hides."""

import json
from typing import Any

RULES = """You are playing Coup with {players} players. The game ends after {max_turns} turns at most.
Each player starts with 2 face-down character cards (their influence) and 2 coins; with 2 players the
starting player gets only 1 coin. The deck holds 15 cards: 3 each of Duke, Assassin, Captain, Ambassador
and Contessa. Coins are public; face-down cards are seen only by their owner.

Players take turns in seat order. On your turn you take exactly one action:
- Income: take 1 coin.
- Foreign Aid: take 2 coins.
- Coup: pay 7 coins; the target loses one influence. If you start your turn with 10 or more coins,
  you must launch a Coup.
A player who loses an influence chooses which of their face-down cards to reveal; revealed cards stay face up.
A player with no face-down cards is out. The last player with influence wins; if the turn limit is
reached first, nobody wins. Talk is allowed but never binding."""


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
    if observation["phase"] == "action":
        parts.append(f"It is turn {observation['turn']} of at most {observation['max_turns']}. {observation['actor']} is choosing an action.")
    elif observation["phase"] == "lose_influence":
        parts.append(f"It is turn {observation['turn']}. {observation['deciding']} must reveal one face-down card.")
    return "\n\n".join(parts)


def answer_request(*, legal_actions: list[Any]) -> str:
    options = "\n".join(f"- {json.dumps(action)}" for action in legal_actions)
    return f"Your answer is one of these actions, written exactly as shown (JSON):\n{options}"


def parse_action(*, answer: str) -> Any:
    """The action as JSON, with a surrounding code fence ignored. A bare word becomes a
    string action (e.g. "pass"). Reading is not checking legality: `step` does that."""
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
        return {str(key).lower(): value.lower() if isinstance(value, str) else value for key, value in action.items()}
    return action.lower() if isinstance(action, str) else action
