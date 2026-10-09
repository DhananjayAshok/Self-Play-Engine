"""Temporary smoke-test game: best-of-N rock paper scissors between two seats.

It exists only to prove the verifiers path end to end (stateless turns, per-seat models,
stop strings, rewards, traces) before the real engine is ported. It is not one of the
engine's games and will be deleted once Coup runs through `SelfPlayEnv`.
"""

import asyncio
import re
from dataclasses import dataclass, field

from self_play.chat_model import STOP_STRING, ChatModel, strip_stop

CHOICES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
FALLBACK = "rock"

SYSTEM_PROMPT = f"""You are Player {{seat}} in a game of rock paper scissors against one opponent, played over {{rounds}} rounds.
Each round both players choose at the same time: rock beats scissors, scissors beats paper, paper beats rock, and equal choices draw.
Whoever wins more rounds wins the game.

Reply with at most two sentences of reasoning, then your choice as <answer>rock</answer>, <answer>paper</answer> or <answer>scissors</answer>, then {STOP_STRING}."""


@dataclass
class SeatRecord:
    replies: list[str] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    invalid: int = 0


@dataclass
class SmokeResult:
    seats: list[SeatRecord]
    round_winners: list[int | None]
    """Per round: the winning seat, or None for a draw."""

    def scores(self) -> list[float]:
        """1 for the seat that won more rounds, 0 for the other, 0.5 each on a tie."""
        wins = [self.round_winners.count(seat) for seat in (0, 1)]
        if wins[0] == wins[1]:
            return [0.5, 0.5]
        return [1.0, 0.0] if wins[0] > wins[1] else [0.0, 1.0]


def parse_choice(*, reply: str) -> str | None:
    """The single choice inside <answer> tags, or None when missing or ambiguous."""
    found = re.findall(r"<answer>\s*(\w+)\s*</answer>", strip_stop(text=reply).lower())
    return found[0] if len(found) == 1 and found[0] in CHOICES else None


def round_winner(*, choices: list[str]) -> int | None:
    if choices[0] == choices[1]:
        return None
    return 0 if BEATS[choices[0]] == choices[1] else 1


def build_messages(*, seat: int, rounds: int, history: list[list[str]]) -> list[dict]:
    """The whole context for one call: the rules plus the game so far, from `seat`'s side."""
    if history:
        lines = [
            f"Round {i + 1}: you chose {played[seat]}, your opponent chose {played[1 - seat]}."
            for i, played in enumerate(history)
        ]
        so_far = "\n".join(lines)
    else:
        so_far = "No rounds have been played yet."
    user = f"{so_far}\n\nRound {len(history) + 1} of {rounds}: make your choice."
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(seat=seat, rounds=rounds)},
        {"role": "user", "content": user},
    ]


async def play_smoke_game(*, models: list[ChatModel], rounds: int) -> SmokeResult:
    """Both seats choose simultaneously each round; an invalid reply plays `FALLBACK`."""
    seats = [SeatRecord(), SeatRecord()]
    history: list[list[str]] = []
    round_winners: list[int | None] = []
    for _ in range(rounds):
        replies = await asyncio.gather(
            *(
                models[seat].complete(messages=build_messages(seat=seat, rounds=rounds, history=history))
                for seat in (0, 1)
            )
        )
        played = []
        for seat, reply in enumerate(replies):
            choice = parse_choice(reply=reply)
            if choice is None:
                seats[seat].invalid += 1
                choice = FALLBACK
            seats[seat].replies.append(reply)
            seats[seat].choices.append(choice)
            played.append(choice)
        history.append(played)
        round_winners.append(round_winner(choices=played))
    return SmokeResult(seats=seats, round_winners=round_winners)
