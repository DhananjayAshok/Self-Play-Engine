"""Coup, base game. See RULES.md for the rules, their sources and every choice made here.

Implemented so far: 2 players; the actions Income, Foreign Aid (not yet blockable) and Coup;
the seeded deal, hidden hands, choosing which influence to lose, elimination and the winner.
Character actions, challenges, blocks and the Exchange come in later steps.

Each decision is its own phase:

    action          ACTION   the actor        talk_rounds from game_config
    lose_influence  ACTION   the loser        reveal one hidden card

Thinking: everyone thinks once at the opening (every game's start); after that only the
next actor thinks, at the start of its turn (`_end_turn`).
"""

from typing import Any

from self_play.game import DialogueMode, DialoguePhase, Game
from self_play.games.coup.prompts import answer_request, parse_action, render_prompt, rules_text

CARDS = ["duke", "assassin", "captain", "ambassador", "contessa"]
COPIES = 3
COUP_COST = 7
FORCED_COUP_COINS = 10
MIN_PLAYERS, MAX_PLAYERS = 2, 2  # widened to 6 once challenges and blocks exist
DEFAULTS = {"max_turns": 100, "talk_rounds": 1}


class CoupGame(Game):
    """game_config: {"seed": int, "players": 2, "max_turns": int >= 1 (default 100),
    "talk_rounds": int >= 0 (default 1, the action phase only)}."""

    name = "coup"

    def initial_state(self) -> dict[str, Any]:
        config = {**DEFAULTS, **self.game_config}
        players, max_turns, talk_rounds = config["players"], config["max_turns"], config["talk_rounds"]
        if not isinstance(players, int) or not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be an integer from {MIN_PLAYERS} to {MAX_PLAYERS}, not {players!r}")
        if not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError(f"max_turns must be a positive integer, not {max_turns!r}")
        if not isinstance(talk_rounds, int) or talk_rounds < 0:
            raise ValueError(f"talk_rounds must be a non-negative integer, not {talk_rounds!r}")
        ids = [f"p{i}" for i in range(1, players + 1)]
        deck = [card for card in CARDS for _ in range(COPIES)]
        self.rng(label="deal", index=0).shuffle(deck)
        hands = {pid: [deck.pop(), deck.pop()] for pid in ids}
        start = self.rng(label="start", index=0).choice(ids)
        coins = {pid: 2 for pid in ids}
        if players == 2:
            coins[start] = 1  # the 2-player rule: the starting player gets only 1 coin
        return {
            "players": ids,
            "hands": hands,          # hidden: each player sees only its own
            "revealed": {pid: [] for pid in ids},
            "coins": coins,
            "deck": deck,            # hidden: only its size is public
            "actor": start,
            "turn": 1,               # the number of the turn being played
            "pending_loss": [],      # players who must reveal a card, in order
            "phase_counter": 0,
            "log": [f"The game begins. {start} plays first."],
            "winner": None,
        }

    # --- helpers -------------------------------------------------------------------

    def setting(self, *, key: str) -> Any:
        return self.game_config.get(key, DEFAULTS[key])

    def alive(self, *, player_id: str) -> bool:
        return bool(self.state["hands"][player_id])

    def living(self) -> list[str]:
        return [pid for pid in self.player_ids if self.alive(player_id=pid)]

    def next_living(self, *, after: str) -> str:
        ids = self.player_ids
        start = ids.index(after)
        for offset in range(1, len(ids) + 1):
            candidate = ids[(start + offset) % len(ids)]
            if self.alive(player_id=candidate):
                return candidate
        raise RuntimeError("no living player")

    @property
    def turn_limit_reached(self) -> bool:
        return self.state["turn"] > self.setting(key="max_turns")

    @property
    def done(self) -> bool:
        return self.state["winner"] is not None or self.turn_limit_reached

    # --- what is happening now -------------------------------------------------

    @property
    def player_ids(self) -> list[str]:
        return list(self.state["players"])

    def _get_dialogue_phase(self) -> DialoguePhase | None:
        if self.done:
            return None
        phase_id = self.state["phase_counter"]
        if self.state["pending_loss"]:
            loser = self.state["pending_loss"][0]
            return DialoguePhase(phase_id=phase_id, phase="lose_influence", mode=DialogueMode.ACTION, eligible=(loser,))
        return DialoguePhase(
            phase_id=phase_id, phase="action", mode=DialogueMode.ACTION, eligible=(self.state["actor"],),
            talk_rounds=self.setting(key="talk_rounds"),
        )

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        phase = self._get_dialogue_phase()
        if phase is None or player_id not in phase.eligible:
            return []
        if phase.phase == "lose_influence":
            # One reveal per distinct hidden card, in hand order.
            return [{"reveal": card} for card in dict.fromkeys(self.state["hands"][player_id])]
        coins = self.state["coins"][player_id]
        targets = [pid for pid in self.living() if pid != player_id]
        coups = [{"action": "coup", "target": pid} for pid in targets]
        if coins >= FORCED_COUP_COINS:
            return coups
        # Income first: the runner's fallback for an invalid answer is the first legal action.
        return [{"action": "income"}, {"action": "foreign_aid"}] + (coups if coins >= COUP_COST else [])

    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Public: coins, revealed cards, hidden-card counts, the deck size, the turn and the
        log. Private: this player's own hidden cards. Never another player's hidden cards or
        the deck's contents."""
        phase = self._get_dialogue_phase()
        state = self.state
        return {
            "you": player_id,
            "your_cards": list(state["hands"][player_id]),
            "players": [
                {
                    "id": pid,
                    "coins": state["coins"][pid],
                    "hidden_cards": len(state["hands"][pid]),
                    "revealed": list(state["revealed"][pid]),
                    "alive": self.alive(player_id=pid),
                }
                for pid in self.player_ids
            ],
            "deck_size": len(state["deck"]),
            "turn": state["turn"],
            "max_turns": self.setting(key="max_turns"),
            "actor": state["actor"],
            "phase": phase.phase if phase else None,
            "deciding": phase.eligible[0] if phase else None,
            "log": list(state["log"]),
            "winner": state["winner"],
        }

    def get_outcome(self) -> dict[str, Any]:
        """The last player with influence gets 1, everyone else 0. At the turn limit nobody wins."""
        winner = self.state["winner"]
        reason = "last_player_standing" if winner else "turn_limit" if self.turn_limit_reached else None
        rewards = {pid: 1.0 if pid == winner else 0.0 for pid in self.player_ids}
        return {"done": self.done, "rewards": rewards, "winner": winner, "termination_reason": reason}

    # --- text for players --------------------------------------------------------

    def get_rules(self, *, observation: dict[str, Any]) -> str:
        return rules_text(players=len(observation["players"]), max_turns=observation["max_turns"])

    def render_observation(self, *, observation: dict[str, Any]) -> str:
        return render_prompt(rules=self.get_rules(observation=observation), observation=observation)

    def render_answer_request(self, *, legal_actions: list[Any]) -> str:
        return answer_request(legal_actions=legal_actions)

    def parse_answer(self, *, player_id: str, answer: str) -> Any:
        return parse_action(answer=answer)

    # --- transitions -------------------------------------------------------------

    def _apply(self, *, player_id: str, action: Any) -> list[str]:
        self.state["phase_counter"] += 1
        if "reveal" in action:
            return self._reveal(player_id=player_id, card=action["reveal"])
        coins = self.state["coins"]
        kind = action["action"]
        if kind == "income":
            coins[player_id] += 1
            events = [f"{player_id} takes Income (+1 coin, now {coins[player_id]})."]
        elif kind == "foreign_aid":
            coins[player_id] += 2
            events = [f"{player_id} takes Foreign Aid (+2 coins, now {coins[player_id]})."]
        else:  # coup
            target = action["target"]
            coins[player_id] -= COUP_COST
            self.state["pending_loss"].append(target)
            events = [f"{player_id} pays {COUP_COST} coins to launch a Coup against {target} (now {coins[player_id]})."]
        self._log(events=events)
        return events + self._end_turn_if_resolved()

    def _reveal(self, *, player_id: str, card: str) -> list[str]:
        self.state["pending_loss"].pop(0)
        hand = self.state["hands"][player_id]
        hand.remove(card)
        self.state["revealed"][player_id].append(card)
        events = [f"{player_id} loses an influence and reveals {card.capitalize()}."]
        if not hand:
            self.state["coins"][player_id] = 0  # coins return to the Treasury
            self.state["pending_loss"] = [pid for pid in self.state["pending_loss"] if pid != player_id]
            events.append(f"{player_id} has no influence left and is out of the game.")
            living = self.living()
            if len(living) == 1:
                self.state["winner"] = living[0]
                events.append(f"{living[0]} is the last player with influence and wins.")
        self._log(events=events)
        return events + self._end_turn_if_resolved()

    def _end_turn_if_resolved(self) -> list[str]:
        """Once nothing is pending, the next living player's turn starts and it thinks first."""
        if self.state["pending_loss"] or self.state["winner"] is not None:
            return []
        self.state["turn"] += 1
        if self.turn_limit_reached:
            events = [f"The turn limit ({self.setting(key='max_turns')}) is reached; nobody wins."]
            self._log(events=events)
            return events
        self.state["actor"] = self.next_living(after=self.state["actor"])
        # Thinking option B (RULES.md): only the next actor. Option A would pass self.living().
        self.request_thinking(player_ids=[self.state["actor"]])
        return []

    def _log(self, *, events: list[str]) -> None:
        self.state["log"].extend(f"Turn {self.state['turn']}: {text}" for text in events)
