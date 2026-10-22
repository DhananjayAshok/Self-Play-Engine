"""What a Chameleon player reads, and how its answers are read. Built only from an
observation, so it can show nothing the observation hides."""

import json
from typing import Any

RULES = """You are playing The Chameleon with {players} players, p1 to p{players} in seat order. One round is one game.
A topic card shows a topic and 16 words. One of the 16 is the secret word. Every player except one knows the
secret word; the odd one out is the Chameleon, who does not know it and must pretend that it does.

1. Clues. Starting with the dealer and going clockwise, every player in turn says exactly one word related to the
   secret word. The Chameleon gives a clue too. Everyone hears every clue. A clue that is too obvious helps the
   Chameleon; a clue that is too vague makes you look like the Chameleon. Repeating a word that was already said
   is allowed.
2. Debate. The players may talk about who the Chameleon is.
3. Vote. All players vote at the same time for the player they think is the Chameleon. You cannot vote for
   yourself. Votes stay hidden until everyone has voted. The player with the most votes is accused. If votes are
   tied, the dealer chooses the accused among the tied players (the dealer may choose itself if it is tied).
4. Result. If the accused is not the Chameleon, the Chameleon escapes and wins. If the accused is the Chameleon,
   it guesses the secret word, one of the 16 words{guesses}. A right guess means the Chameleon escapes. A wrong
   guess means the Chameleon is caught and the other players win.

Rewards: the Chameleon escapes undetected: Chameleon 1, others 0. The Chameleon is caught but guesses right:
Chameleon 0.5, others 0. The Chameleon is caught and guesses wrong: Chameleon 0, others 1. Talk is allowed but
never binding."""

CLUE_LINES = "In the table talk, each player's first line is its clue."

THREE_PLAYER_GUESSES = " (with 3 players it gets two guesses and stops at the first right one)"


def rules_text(*, players: int) -> str:
    return RULES.format(players=players, guesses=THREE_PLAYER_GUESSES if players == 3 else "")


def _clue_order(*, observation: dict[str, Any]) -> str:
    return ", ".join(observation["clue_order"])


def _votes_text(*, observation: dict[str, Any]) -> str:
    return ", ".join(f"{voter} voted for {target}" for voter, target in observation["votes"])


def render_prompt(*, rules: str, observation: dict[str, Any]) -> str:
    you = observation["you"]
    parts = [rules, f"You are {you}. The dealer is {observation['dealer']}.",
             f"The topic is {observation['topic']}. The 16 words on the topic card: {', '.join(observation['words'])}."]
    if observation["is_chameleon"]:
        parts.append("You are the Chameleon. You do not know the secret word; it is one of the 16 words.")
    else:
        parts.append(f"You are not the Chameleon. The secret word is: {observation['secret_word']}. "
                     "The Chameleon does not know it.")
    parts.append("What has happened so far:\n" + "\n".join(f"- {line}" for line in observation["log"]))
    if observation["phase"] is not None:
        parts.append(situation_text(observation=observation))
    return "\n\n".join(parts)


def situation_text(*, observation: dict[str, Any]) -> str:
    """What is going on now and, for the players being asked, what exactly they decide."""
    phase = observation["phase"]
    asked = observation["you"] in observation["deciding"] and not observation["thinking"]  # notes come first
    if phase == "clues":
        text = (f"Clue round, in this order: {_clue_order(observation=observation)}. Each player says exactly one word.")
        if asked:
            if observation["is_chameleon"]:
                text += (" Your decision now: say exactly one word related to the topic that fits the secret word "
                         "you cannot see, without giving away that you are the Chameleon.")
            else:
                text += (" Your decision now: say exactly one word related to the secret word. Do not say the "
                         "secret word itself, and do not make it so obvious that the Chameleon can guess it.")
        return text
    if phase == "debate":
        text = f"Debate: the clues are in and the players may talk about who the Chameleon is. {CLUE_LINES}"
        if asked:
            text += " Your decision now: say something to the table, or stay silent."
        return text
    if phase == "vote":
        text = f"Vote: everyone votes at the same time for the player they think is the Chameleon. {CLUE_LINES}"
        if asked:
            text += " Your decision now: vote for another player. You cannot vote for yourself."
        return text
    if phase == "tiebreak":
        text = (f"The votes were: {_votes_text(observation=observation)}. Tied for the most votes: "
                f"{', '.join(observation['tied'])}. {CLUE_LINES}")
        if asked:
            text += " Your decision now: you are the dealer; choose which tied player is accused."
        return text
    left, wrong = observation["guesses_left"], observation["wrong_guesses"]
    text = f"{observation['accused']} was accused and is the Chameleon, and now guesses the secret word."
    if wrong:
        text += f" Wrong guesses so far: {', '.join(wrong)}."
    if asked:
        text += f" Your decision now: guess the secret word, one of the words you may still pick. Guesses left: {left}."
    return text


PLACEHOLDERS = {"vote": ("pX", "the player you choose"), "choose": ("pX", "the player you choose"),
                "guess": ("WORD", "one of the words you may still pick")}
"""The example in an answer request never shows a real choice, so a model cannot copy it as its answer."""


def answer_request(*, legal_actions: list[Any]) -> str:
    options = "\n".join(f"- {json.dumps(action)}" for action in legal_actions)
    first = legal_actions[0]
    key = next(iter(first)) if isinstance(first, dict) else None
    placeholder, meaning = PLACEHOLDERS.get(key, (None, None))
    if placeholder is None:
        example = json.dumps(first)
        ending = ""
    else:
        example = json.dumps({key: placeholder})
        ending = f", with {placeholder} replaced by {meaning}."
    return (f"Your answer is one of these actions, written exactly as shown (JSON):\n{options}\n"
            f"Write your answer inside the answer tags, for example: <answer>{example}</answer>{ending}")


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
        return {str(key).lower(): value.lower() if isinstance(value, str) else value for key, value in action.items()}
    return action.lower() if isinstance(action, str) else action
