"""Building model prompts and reading replies. Game-agnostic: every game-specific word
comes from the Game (`render_observation`, `render_answer_request`).

Every call is the whole context (system + one user message), never a growing chat.

Reply format asked of the model, for every kind of call:

    <thinking>private scratch work</thinking>
    <answer>the answer</answer>
    [STOP]
"""

import re

from self_play.chat_model import STOP_STRING, strip_stop

NO_CHANGE = "NO CHANGE"
"""A thinking turn's answer that keeps the current notes."""

SILENCE = "SILENCE"
"""A talk turn's answer that says nothing."""

SYSTEM_PROMPT = f"""You are a player in a game. Every time you are asked something, reply in exactly this format:
<thinking>your private reasoning</thinking>
<answer>your answer</answer>
{STOP_STRING}

Your thinking is scratch work: nobody else sees it, and you will not see it again.
Your notes are your only memory between turns. You rewrite them only in thinking turns, and they are shown to you every time you are asked something."""

THINK_INSTRUCTION = f"""This is a thinking turn. Write your complete new notes inside <answer>: what you believe, your plan, and anything you would otherwise forget. They replace your current notes entirely and are shown to you at every turn until your next thinking turn.
To keep your current notes exactly as they are, answer {NO_CHANGE}."""

TALK_INSTRUCTION = f"""Before you decide, you may say something to the table. Put what you say inside <answer>; everyone hears it, and it is not binding.
To say nothing, answer {SILENCE}."""

_THINKING = re.compile(r"<thinking>(.*?)</thinking>", re.S | re.I)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def build_messages(
    *, game_text: str, notes: str, talk: list[dict], instruction: str, feedback: str | None = None
) -> list[dict]:
    """One call's whole context: the game's text, the player's notes, the table talk it
    may see, and what to do now. `feedback` explains why the previous answer was rejected."""
    parts = [game_text, f"Your notes:\n{notes}" if notes else "Your notes: (none yet)"]
    if talk:
        lines = "\n".join(f"- {entry['player_id']} said: {entry['content']}" for entry in talk)
        parts.append(f"Table talk:\n{lines}")
    parts.append(instruction)
    if feedback:
        parts.append(f"Your previous answer was rejected: {feedback}. Answer again.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]


def parse_reply(*, reply: str) -> tuple[str | None, str | None]:
    """(thinking, answer) from a reply, cut at the stop string. Either is None when its
    tags are missing; a missing answer is never guessed from the rest of the text."""
    text = strip_stop(text=reply)
    thinking, answer = _THINKING.search(text), _ANSWER.search(text)
    return (thinking.group(1).strip() if thinking else None, answer.group(1).strip() if answer else None)


def is_keyword(*, answer: str | None, keyword: str) -> bool:
    """True when the answer is the keyword, ignoring case and surrounding punctuation."""
    return answer is not None and answer.strip().strip(".!\"'`*").strip().upper() == keyword
