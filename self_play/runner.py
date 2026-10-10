"""The runner drives one game: who is asked when, model calls, talk, thoughtpads, invalid
answers, and the record. It contains no game rules: it reads the open `DialoguePhase` and
schedules by its `mode` only.

Division of labour:
    Game    -> WHO MAY act, WHAT is legal, and what happens.
    Runner  -> WHO IS ASKED FIRST, talk, notes, fallbacks, and the record.
    Model   -> WHAT each seat says and answers (one `ChatModel` per player).

The record is the single memory of the game besides the Game itself. Entry kinds:
    phase    a new dialogue phase opened
    call     one model call: who, which kind (think, talk, answer), messages, reply, parse
    message  table talk
    action   one attempt to act, accepted or not, and where it came from
    event    something the game made public
A player's notes (thoughtpad) are the `notes` of its latest think call.

Failures: a model call that fails raises and stops the game (see `ModelCallFailed`). An
answer that cannot be read or is not legal is not a failure: after `max_retries` retries
with the reason, the player's first legal action is played and counted as invalid.

Checkpoints (when `checkpoint_path` is given): the snapshot holds the game as it was when
the open phase began, plus a journal of every model call completed since. It is rewritten
after every call, so a crash loses at most the call in flight. A new Runner on the same
path resumes: it restores the phase start and replays the phase with the journal's replies
instead of calling the model (each replayed prompt must match the saved one exactly), then
continues live. A finished game's snapshot holds its result, so it is never played again.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from self_play.chat_model import ChatModel
from self_play.checkpoint import load_snapshot, save_snapshot
from self_play.game import THINK, DialogueMode, DialoguePhase, Game, StepResult
from self_play.prompting import (
    NO_CHANGE,
    SILENCE,
    TALK_INSTRUCTION,
    THINK_INSTRUCTION,
    build_messages,
    is_keyword,
    parse_reply,
)

TALK_WINDOW = 40
"""How many of the latest table messages a talk or answer call shows."""


class Runner:
    def __init__(
        self,
        *,
        game: Game,
        models: dict[str, ChatModel],
        max_retries: int = 0,
        talk_window: int = TALK_WINDOW,
        checkpoint_path: str | Path | None = None,
    ):
        missing, unknown = set(game.player_ids) - set(models), set(models) - set(game.player_ids)
        if missing or unknown:
            raise ValueError(f"models must match the game's players: missing {sorted(missing)}, unknown {sorted(unknown)}")
        self.game = game
        self.models = dict(models)
        self.max_retries = max_retries
        self.talk_window = talk_window
        self.record: list[dict[str, Any]] = []
        self.invalid = {player_id: 0 for player_id in game.player_ids}
        """Per player: answers that were unreadable or illegal and ended in the fallback,
        plus thinking turns without a readable answer."""
        self._last_phase_id: int | None = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.finished_result: dict[str, Any] | None = None
        """Set once the game is over (or when resuming a finished game)."""
        self.resumed = False
        """Whether this runner picked up a saved snapshot."""
        self._phase_start: dict[str, Any] | None = None
        self._journal: list[dict[str, Any]] = []
        self._replay: list[dict[str, Any]] = []
        if self.checkpoint_path is not None:
            snapshot = load_snapshot(path=self.checkpoint_path)
            if snapshot is not None:
                self._restore(snapshot=snapshot)

    # ------------------------------------------------------------------ public

    async def run(self) -> dict[str, Any]:
        """Play until the game is over; return its result."""
        if self.finished_result is not None:
            return self.finished_result
        while (phase := self.game.get_dialogue_phase()) is not None:
            self._begin_phase()
            await self.run_phase(phase=phase)
            if self._replay:
                raise RuntimeError(f"the checkpoint holds {len(self._replay)} call(s) this game never made again")
            if self.game.get_dialogue_phase() == phase:
                raise RuntimeError(f"phase {phase.to_dict()} is still open after everyone in it acted")
        self.finished_result = self.result()
        self._save()
        return self.finished_result

    def result(self) -> dict[str, Any]:
        return {
            "outcome": self.game.get_outcome(),
            "step_rewards": self.game.get_step_rewards(),
            "invalid": dict(self.invalid),
        }

    async def run_phase(self, *, phase: DialoguePhase) -> None:
        """Run the open phase once: its talk, then its actions."""
        if phase.phase_id != self._last_phase_id:
            self._last_phase_id = phase.phase_id
            self.record.append({"kind": "phase", **phase.to_dict(), "n_steps": self.game.n_steps})
        if phase.mode is DialogueMode.THINKING:
            await self._run_thinking(phase=phase)
        elif phase.mode is DialogueMode.DISCUSSION:
            for _ in range(phase.talk_rounds):
                for player_id in phase.eligible:
                    await self._talk(phase=phase, player_id=player_id)
            for player_id in phase.eligible:
                if self._still_eligible(phase=phase, player_id=player_id):
                    await self._act(phase=phase, player_id=player_id)
        elif phase.mode is DialogueMode.SIMULTANEOUS:
            await self._run_simultaneous(phase=phase)
        else:  # ACTION and RESPONSE_WINDOW: in order, stopping as soon as the game closes the phase
            for player_id in phase.eligible:
                if not self._still_eligible(phase=phase, player_id=player_id):
                    continue
                if not self._is_forced(player_id=player_id):
                    for _ in range(phase.talk_rounds):
                        await self._talk(phase=phase, player_id=player_id)
                await self._act(phase=phase, player_id=player_id)

    def notes(self, *, player_id: str) -> str:
        """The player's current thoughtpad: the notes of its latest think call."""
        for entry in reversed(self.record):
            if entry["kind"] == "call" and entry["call"] == "think" and entry["player_id"] == player_id:
                return entry["notes"]
        return ""

    def call_step_ids(self) -> list[int | None]:
        """For each call entry, in record order: the id of the step it led to, i.e. the
        caller's next accepted action. None if the caller never acted again."""
        out: list[int | None] = []
        next_step: dict[str, int | None] = {}
        for entry in reversed(self.record):
            if entry["kind"] == "action" and entry["accepted"]:
                next_step[entry["player_id"]] = entry["step_id"]
            elif entry["kind"] == "call":
                out.append(next_step.get(entry["player_id"]))
        return out[::-1]

    # ------------------------------------------------------------- checkpoints

    def _begin_phase(self) -> None:
        """Remember the game as the open phase begins; the journal restarts with it."""
        self._phase_start = {
            "game": self.game.to_dict(),
            "record_len": len(self.record),
            "invalid": dict(self.invalid),
            "last_phase_id": self._last_phase_id,
        }
        self._journal = []
        self._save()

    def _save(self) -> None:
        if self.checkpoint_path is None:
            return
        snapshot: dict[str, Any] = {"game": self.game.name, "game_config": self.game.game_config, "record": self.record}
        if self.finished_result is not None:
            snapshot |= {"finished": True, "result": self.finished_result, "final_game": self.game.to_dict(),
                         "invalid": self.invalid}
        else:
            snapshot |= {"finished": False, "phase_start": self._phase_start, "journal": self._journal}
        save_snapshot(path=self.checkpoint_path, snapshot=snapshot)

    def _restore(self, *, snapshot: dict[str, Any]) -> None:
        same_config = json.dumps(snapshot["game_config"], sort_keys=True) == json.dumps(self.game.game_config, sort_keys=True)
        if snapshot["game"] != self.game.name or not same_config:
            raise ValueError(f"{self.checkpoint_path} holds a different game: {snapshot['game']} {snapshot['game_config']}")
        game_class = type(self.game)
        self.resumed = True
        if snapshot["finished"]:
            self.game = game_class.from_dict(data=snapshot["final_game"])
            self.record = snapshot["record"]
            self.invalid = dict(snapshot["invalid"])
            self.finished_result = snapshot["result"]
            return
        start = snapshot["phase_start"]
        self.game = game_class.from_dict(data=start["game"])
        self.record = snapshot["record"][: start["record_len"]]
        self.invalid = dict(start["invalid"])
        self._last_phase_id = start["last_phase_id"]
        self._replay = list(snapshot["journal"])

    def _take_replay(self, *, player_id: str, kind: str, messages: list[dict]) -> dict[str, Any] | None:
        """The saved reply for this call when resuming, else None (call the model)."""
        for index, entry in enumerate(self._replay):
            if entry["player_id"] == player_id and entry["call"] == kind:
                if entry["messages"] != messages:
                    raise RuntimeError(f"resuming {self.checkpoint_path}: a {kind} prompt for {player_id} differs from the saved one")
                return dict(self._replay.pop(index))
        return None

    # -------------------------------------------------------------- the modes

    async def _run_thinking(self, *, phase: DialoguePhase) -> None:
        """Every eligible player rewrites its notes, independently and at once."""
        calls = await asyncio.gather(*(
            self._call(phase=phase, player_id=player_id, kind="think", instruction=THINK_INSTRUCTION)
            for player_id in phase.eligible
        ))
        for player_id, entry in zip(phase.eligible, calls, strict=True):
            answer = entry["answer"]
            entry["invalid"] = answer is None
            if answer is None or is_keyword(answer=answer, keyword=NO_CHANGE):
                entry["notes"] = self.notes(player_id=player_id)
                self.invalid[player_id] += answer is None
            else:
                entry["notes"] = answer
            self.record.append(entry)
            self._step_or_raise(phase=phase, player_id=player_id, action=THINK, source="think")

    async def _run_simultaneous(self, *, phase: DialoguePhase) -> None:
        """Collect every eligible player's answer BEFORE any step, so nobody sees another's choice."""
        asked = []
        for player_id in phase.eligible:
            if not self._is_forced(player_id=player_id):
                for _ in range(phase.talk_rounds):
                    await self._talk(phase=phase, player_id=player_id)
                asked.append(player_id)
        calls = await asyncio.gather(*(self._ask(phase=phase, player_id=player_id) for player_id in asked))
        first = dict(zip(asked, calls, strict=True))
        for player_id in asked:
            self.record.append(first[player_id])
        for player_id in phase.eligible:
            if self._still_eligible(phase=phase, player_id=player_id):
                await self._act(phase=phase, player_id=player_id, first_call=first.get(player_id))

    # ------------------------------------------------------------ one decision

    def _still_eligible(self, *, phase: DialoguePhase, player_id: str) -> bool:
        current = self.game.get_dialogue_phase()
        return current is not None and current.phase_id == phase.phase_id and player_id in current.eligible

    def _is_forced(self, *, player_id: str) -> bool:
        return len(self.game.get_legal_actions(player_id=player_id)) == 1

    def _talk_seen(self, *, player_id: str, kind: str) -> list[dict[str, Any]]:
        """Table talk shown in a call: since the player's last thinking turn for a think
        call, otherwise the latest `talk_window` messages. All talk is public."""
        if kind == "think":
            start = 0
            for index, entry in enumerate(self.record):
                if entry["kind"] == "call" and entry["call"] == "think" and entry["player_id"] == player_id:
                    start = index
            return [e for e in self.record[start:] if e["kind"] == "message"]
        return [e for e in self.record if e["kind"] == "message"][-self.talk_window:]

    async def _call(
        self, *, phase: DialoguePhase, player_id: str, kind: str, instruction: str, feedback: str | None = None
    ) -> dict[str, Any]:
        """One model call (or its saved reply when resuming). Returns its record entry; the
        caller appends it, so calls made at once land in the record in a fixed order. Every
        completed call is journaled and checkpointed at once."""
        observation = self.game.get_observation(player_id=player_id)
        messages = build_messages(
            game_text=self.game.render_observation(observation=observation),
            notes=self.notes(player_id=player_id),
            talk=self._talk_seen(player_id=player_id, kind=kind),
            instruction=instruction,
            feedback=feedback,
        )
        entry = self._take_replay(player_id=player_id, kind=kind, messages=messages)
        if entry is None:
            reply = await self.models[player_id].complete(messages=messages)
            thinking, answer = parse_reply(reply=reply)
            entry = {
                "kind": "call",
                "call": kind,
                "player_id": player_id,
                "phase_id": phase.phase_id,
                "n_steps": self.game.n_steps,
                "messages": messages,
                "reply": reply,
                "thinking": thinking,
                "answer": answer,
            }
        self._journal.append(dict(entry))
        self._save()
        return entry

    async def _talk(self, *, phase: DialoguePhase, player_id: str) -> None:
        entry = await self._call(phase=phase, player_id=player_id, kind="talk", instruction=TALK_INSTRUCTION)
        self.record.append(entry)
        answer = entry["answer"]
        if answer and not is_keyword(answer=answer, keyword=SILENCE):
            self.record.append({"kind": "message", "player_id": player_id, "phase_id": phase.phase_id, "content": answer})

    async def _ask(self, *, phase: DialoguePhase, player_id: str, feedback: str | None = None) -> dict[str, Any]:
        legal = self.game.get_legal_actions(player_id=player_id)
        instruction = self.game.render_answer_request(legal_actions=legal)
        return await self._call(phase=phase, player_id=player_id, kind="answer", instruction=instruction, feedback=feedback)

    async def _act(self, *, phase: DialoguePhase, player_id: str, first_call: dict[str, Any] | None = None) -> None:
        """Ask (unless already asked), try the answer, retry with the reason, then fall back
        to the first legal action and count it as invalid."""
        legal = self.game.get_legal_actions(player_id=player_id)
        if len(legal) == 1:
            self._step_or_raise(phase=phase, player_id=player_id, action=legal[0], source="forced")
            return
        entry = first_call
        if entry is None:
            entry = await self._ask(phase=phase, player_id=player_id)
            self.record.append(entry)
        for attempt in range(self.max_retries + 1):
            error = self._try_answer(phase=phase, player_id=player_id, answer=entry["answer"])
            if error is None:
                return
            if attempt < self.max_retries:
                entry = await self._ask(phase=phase, player_id=player_id, feedback=error)
                self.record.append(entry)
        self.invalid[player_id] += 1
        fallback = self.game.get_legal_actions(player_id=player_id)[0]
        self._step_or_raise(phase=phase, player_id=player_id, action=fallback, source="fallback")

    def _try_answer(self, *, phase: DialoguePhase, player_id: str, answer: str | None) -> str | None:
        """Parse and step one answer. None if accepted, otherwise why not."""
        if answer is None:
            error = "no <answer>...</answer> was found in the reply"
        else:
            try:
                action = self.game.parse_answer(player_id=player_id, answer=answer)
            except ValueError as reason:
                error = f"could not read the answer: {reason}"
            else:
                return self._step(phase=phase, player_id=player_id, action=action, source="model").error
        self.record.append({
            "kind": "action", "player_id": player_id, "phase_id": phase.phase_id, "action": None,
            "source": "model", "accepted": False, "error": error, "step_id": None,
        })
        return error

    def _step(self, *, phase: DialoguePhase, player_id: str, action: Any, source: str) -> StepResult:
        result = self.game.step(player_id=player_id, action=action)
        self.record.append({
            "kind": "action", "player_id": player_id, "phase_id": phase.phase_id, "action": action,
            "source": source, "accepted": result.accepted, "error": result.error,
            "step_id": self.game.n_steps - 1 if result.accepted else None,
        })
        for text in result.public_events:
            self.record.append({"kind": "event", "phase_id": phase.phase_id, "content": text})
        return result

    def _step_or_raise(self, *, phase: DialoguePhase, player_id: str, action: Any, source: str) -> None:
        result = self._step(phase=phase, player_id=player_id, action=action, source=source)
        if not result.accepted:
            raise RuntimeError(f"{source} action {action!r} for {player_id} was rejected: {result.error}")
