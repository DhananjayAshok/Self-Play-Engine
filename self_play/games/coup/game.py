"""Coup, base game. See RULES.md for the rules, their sources and every choice made here.

2 to 6 players; every action, the challenge of a claim, blocks, the challenge of a block,
choosing which influence to lose, the Exchange, elimination and the winner.

Each decision is its own phase:

    action           ACTION           the actor            talk_rounds
    challenge_action RESPONSE_WINDOW  everyone else        window_talk_rounds
    block_action     RESPONSE_WINDOW  who may block        window_talk_rounds
    challenge_block  RESPONSE_WINDOW  all but the blocker  window_talk_rounds
    lose_influence   ACTION           the loser            0
    exchange         ACTION           the Ambassador       0

A turn is `state["turn_ctx"]`: the declared action and its `stage`, the next thing to do.
Window stages wait for answers; a pass removes the player from `responders` (same phase),
anything else moves the stage on (a new phase). Losses queue in `pending_loss` and are
revealed before the stage goes on, so one turn can cost a player two influences.

Turns go clockwise (seat order) from the seeded starting player, skipping eliminated players.
Thinking: everyone thinks once at the opening (every game's start); after that game_config
"thinking" decides who thinks when a turn ends: "actor" (the next actor only) or "all".
"""

from itertools import combinations
from typing import Any

from self_play.game import DialogueMode, DialoguePhase, Game
from self_play.games.coup.prompts import answer_request, parse_action, render_prompt, rules_text

CARDS = ["duke", "assassin", "captain", "ambassador", "contessa"]
COPIES = 3
COUP_COST = 7
ASSASSINATE_COST = 3
FORCED_COUP_COINS = 10
MIN_PLAYERS, MAX_PLAYERS = 2, 6
THINKING = ("actor", "all")
DEFAULTS = {"max_turns": 100, "talk_rounds": 1, "window_talk_rounds": 0, "thinking": "actor"}

CLAIMS = {"tax": "duke", "assassinate": "assassin", "steal": "captain", "exchange": "ambassador"}
"""The character each action claims. Income, Foreign Aid and Coup claim none."""
BLOCKS = {"foreign_aid": ["duke"], "steal": ["captain", "ambassador"], "assassinate": ["contessa"]}
"""The characters that may block each blockable action."""
WINDOWS = ("challenge_action", "block_action", "challenge_block")
ACTION_NAMES = {"income": "Income", "foreign_aid": "Foreign Aid", "coup": "Coup", "tax": "Tax",
                "assassinate": "Assassinate", "steal": "Steal", "exchange": "Exchange"}


def check_count(*, key: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, not {value!r}")


def describe(*, action: dict[str, Any]) -> str:
    """'Steal from p3 (claims Captain)' and the like."""
    kind = action["action"]
    text = ACTION_NAMES[kind]
    if "target" in action:
        text += f" {'from' if kind == 'steal' else 'against'} {action['target']}"
    if kind in CLAIMS:
        text += f" (claims {CLAIMS[kind].capitalize()})"
    return text


class CoupGame(Game):
    """game_config: {"seed": int, "players": 2..6, "max_turns": int >= 1 (default 100),
    "talk_rounds": int >= 0 (default 1, the action phase), "window_talk_rounds": int >= 0
    (default 0, challenge and block windows), "thinking": "actor" (default) or "all"}."""

    name = "coup"

    def initial_state(self) -> dict[str, Any]:
        config = {**DEFAULTS, **self.game_config}
        players = config["players"]
        if not isinstance(players, int) or not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be an integer from {MIN_PLAYERS} to {MAX_PLAYERS}, not {players!r}")
        check_count(key="max_turns", value=config["max_turns"], minimum=1)
        check_count(key="talk_rounds", value=config["talk_rounds"], minimum=0)
        check_count(key="window_talk_rounds", value=config["window_talk_rounds"], minimum=0)
        if config["thinking"] not in THINKING:
            raise ValueError(f"thinking must be one of {', '.join(THINKING)}, not {config['thinking']!r}")
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
            "deck": deck,            # hidden: only its size is public; the top card is the last
            "shuffles": 0,           # deck shuffles so far, the index of the next shuffle's draw
            "actor": start,
            "turn": 1,               # the number of the turn being played
            "turn_ctx": None,        # the declared action being resolved, see the module docstring
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

    def living_clockwise(self, *, after: str) -> list[str]:
        """Every living player except `after`, clockwise from it."""
        ids = self.player_ids
        start = ids.index(after)
        order = [ids[(start + offset) % len(ids)] for offset in range(1, len(ids))]
        return [pid for pid in order if self.alive(player_id=pid)]

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
        ctx = self.state["turn_ctx"]
        if ctx is None:
            return DialoguePhase(
                phase_id=phase_id, phase="action", mode=DialogueMode.ACTION, eligible=(self.state["actor"],),
                talk_rounds=self.setting(key="talk_rounds"),
            )
        if ctx["stage"] in WINDOWS:
            return DialoguePhase(
                phase_id=phase_id, phase=ctx["stage"], mode=DialogueMode.RESPONSE_WINDOW,
                eligible=tuple(ctx["responders"]), talk_rounds=self.setting(key="window_talk_rounds"),
            )
        # The only stage that waits for an answer outside a window.
        return DialoguePhase(phase_id=phase_id, phase="exchange", mode=DialogueMode.ACTION, eligible=(ctx["actor"],))

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        # The first listed action is the safe one: the runner's fallback for an invalid answer.
        phase = self._get_dialogue_phase()
        if phase is None or player_id not in phase.eligible:
            return []
        ctx = self.state["turn_ctx"]
        if phase.phase == "lose_influence":
            return [{"reveal": card} for card in dict.fromkeys(self.state["hands"][player_id])]
        if phase.phase in ("challenge_action", "challenge_block"):
            return ["pass", "challenge"]
        if phase.phase == "block_action":
            return ["pass"] + [{"block": card} for card in BLOCKS[ctx["action"]["action"]]]
        if phase.phase == "exchange":
            return self._exchange_options(player_id=player_id)
        coins = self.state["coins"][player_id]
        targets = [pid for pid in self.living() if pid != player_id]
        coups = [{"action": "coup", "target": pid} for pid in targets]
        if coins >= FORCED_COUP_COINS:
            return coups
        actions = [{"action": "income"}, {"action": "foreign_aid"}, {"action": "tax"}, {"action": "exchange"}]
        actions += [{"action": "steal", "target": pid} for pid in targets]
        if coins >= ASSASSINATE_COST:
            actions += [{"action": "assassinate", "target": pid} for pid in targets]
        return actions + (coups if coins >= COUP_COST else [])

    def _exchange_options(self, *, player_id: str) -> list[dict[str, Any]]:
        """Every distinct set of as many cards as the player holds, from its hand and the two
        drawn cards; keeping the current hand comes first."""
        hand = sorted(self.state["hands"][player_id])
        pool = hand + self.state["turn_ctx"]["drawn"]
        options = sorted({tuple(sorted(keep)) for keep in combinations(pool, len(hand))} - {tuple(hand)})
        return [{"keep": hand}] + [{"keep": list(keep)} for keep in options]

    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Public: coins, revealed cards, hidden-card counts, the deck size, the turn, the
        action being resolved (who declared what, any block) and the log. Private: this
        player's own hidden cards and, during its own Exchange, the two drawn cards. Never
        another player's hidden cards, cards drawn by someone else, or the deck's contents."""
        phase = self._get_dialogue_phase()
        state = self.state
        ctx = state["turn_ctx"]
        pending = None
        if ctx is not None:
            pending = {"actor": ctx["actor"], "action": dict(ctx["action"]),
                       "block": dict(ctx["block"]) if ctx["block"] else None}
        drawn = list(ctx["drawn"]) if ctx is not None and ctx["drawn"] and ctx["actor"] == player_id else None
        return {
            "you": player_id,
            "your_cards": list(state["hands"][player_id]),
            "your_drawn_cards": drawn,
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
            "pending": pending,
            "phase": phase.phase if phase else None,
            "deciding": list(phase.eligible) if phase else [],
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
        ctx = self.state["turn_ctx"]
        events: list[str] = []
        if action == "pass":
            ctx["responders"].remove(player_id)
            events.append(f"{player_id} passes.")
            if ctx["responders"]:
                self._log(events=events)
                return events  # the window stays open: same phase, one player fewer
            ctx["stage"] = {"challenge_action": "block_or_resolve", "block_action": "resolve",
                            "challenge_block": "end"}[ctx["stage"]]
        elif action == "challenge":
            events += self._challenge(challenger=player_id)
        elif "reveal" in action:
            events += self._reveal(player_id=player_id, card=action["reveal"])
        elif "block" in action:
            ctx["block"] = {"by": player_id, "card": action["block"]}
            ctx["stage"] = "challenge_block"
            ctx["responders"] = self.living_clockwise(after=player_id)
            events.append(f"{player_id} blocks, claiming {action['block'].capitalize()}.")
        elif "keep" in action:
            events += self._keep(player_id=player_id, keep=action["keep"])
        else:
            events += self._declare(player_id=player_id, action=action)
        self.state["phase_counter"] += 1
        self._log(events=events)
        return events + self._advance()

    def _declare(self, *, player_id: str, action: dict[str, Any]) -> list[str]:
        kind = action["action"]
        cost = {"coup": COUP_COST, "assassinate": ASSASSINATE_COST}.get(kind, 0)
        self.state["coins"][player_id] -= cost  # paid now; refunded only if the claim is caught
        ctx = {"actor": player_id, "action": dict(action), "cost": cost, "stage": "block_or_resolve",
               "responders": [], "block": None, "drawn": []}
        self.state["turn_ctx"] = ctx
        if kind in CLAIMS:
            ctx["stage"] = "challenge_action"
            ctx["responders"] = self.living_clockwise(after=player_id)
        paid = f", paying {cost} coins" if cost else ""
        return [f"{player_id} declares {describe(action=action)}{paid}."]

    def _challenge(self, *, challenger: str) -> list[str]:
        """The claimant shows the card if it holds it (it always does, RULES.md), shuffles it
        back and draws a replacement; then the loser of the challenge loses one influence."""
        ctx = self.state["turn_ctx"]
        if ctx["stage"] == "challenge_action":
            claimant, card = ctx["actor"], CLAIMS[ctx["action"]["action"]]
        else:
            claimant, card = ctx["block"]["by"], ctx["block"]["card"]
        events = [f"{challenger} challenges {claimant}'s claim to hold {card.capitalize()}."]
        if card in self.state["hands"][claimant]:
            self._replace(player_id=claimant, card=card)
            events.append(f"{claimant} shows {card.capitalize()}, shuffles it into the deck and draws a new card; "
                          f"{challenger} loses the challenge.")
            self.state["pending_loss"].append(challenger)
            # A proven action goes on to any block; a proven block stands.
            ctx["stage"] = "block_or_resolve" if ctx["stage"] == "challenge_action" else "end"
        else:
            events.append(f"{claimant} cannot show {card.capitalize()} and loses the challenge.")
            self.state["pending_loss"].append(claimant)
            if ctx["stage"] == "challenge_action":
                self.state["coins"][claimant] += ctx["cost"]  # a caught claim refunds its cost
                if ctx["cost"]:
                    events.append(f"{claimant}'s {ctx['cost']} coins are refunded; the action fails.")
                else:
                    events.append("The action fails.")
                ctx["stage"] = "end"
            else:
                events.append("The block fails.")
                ctx["stage"] = "resolve"
        return events

    def _replace(self, *, player_id: str, card: str) -> None:
        hand, deck = self.state["hands"][player_id], self.state["deck"]
        hand.remove(card)
        deck.append(card)
        self._shuffle()
        hand.append(deck.pop())

    def _shuffle(self) -> None:
        self.rng(label="shuffle", index=self.state["shuffles"]).shuffle(self.state["deck"])
        self.state["shuffles"] += 1

    def _keep(self, *, player_id: str, keep: list[str]) -> list[str]:
        ctx = self.state["turn_ctx"]
        pool = self.state["hands"][player_id] + ctx["drawn"]
        for card in keep:
            pool.remove(card)
        self.state["hands"][player_id] = list(keep)
        self.state["deck"].extend(pool)
        self._shuffle()
        ctx["drawn"] = []
        ctx["stage"] = "end"
        return [f"{player_id} exchanges cards and returns two to the deck."]

    def _reveal(self, *, player_id: str, card: str) -> list[str]:
        self.state["pending_loss"].pop(0)
        hand = self.state["hands"][player_id]
        hand.remove(card)
        self.state["revealed"][player_id].append(card)
        events = [f"{player_id} loses an influence and reveals {card.capitalize()}."]
        if not hand:
            self.state["coins"][player_id] = 0  # coins return to the Treasury
            self.state["pending_loss"] = [pid for pid in self.state["pending_loss"] if pid != player_id]
            ctx = self.state["turn_ctx"]
            if ctx is not None and player_id in ctx["responders"]:
                ctx["responders"].remove(player_id)
            events.append(f"{player_id} has no influence left and is out of the game.")
            living = self.living()
            if len(living) == 1:
                self.state["winner"] = living[0]
                events.append(f"{living[0]} is the last player with influence and wins.")
        return events

    def _advance(self) -> list[str]:
        """Run the turn's stages that need no answer, until one does or the turn ends."""
        events: list[str] = []
        while self.state["winner"] is None and not self.state["pending_loss"]:
            ctx = self.state["turn_ctx"]
            if ctx is None:
                break
            stage = ctx["stage"]
            if stage in WINDOWS or stage == "exchange":
                break
            if stage == "block_or_resolve":
                kind, target = ctx["action"]["action"], ctx["action"].get("target")
                blockers = []
                if kind == "foreign_aid":
                    blockers = self.living_clockwise(after=ctx["actor"])
                elif kind in BLOCKS and self.alive(player_id=target):
                    blockers = [target]
                ctx["stage"], ctx["responders"] = ("block_action", blockers) if blockers else ("resolve", [])
                if blockers:
                    self.state["phase_counter"] += 1
            elif stage == "resolve":
                resolved = self._resolve()
                self._log(events=resolved)
                events += resolved
            else:  # end
                self.state["turn_ctx"] = None
                ended = self._end_turn()
                self._log(events=ended)
                events += ended
        return events

    def _resolve(self) -> list[str]:
        ctx = self.state["turn_ctx"]
        actor, kind, target = ctx["actor"], ctx["action"]["action"], ctx["action"].get("target")
        coins = self.state["coins"]
        ctx["stage"] = "end"
        if kind == "income":
            coins[actor] += 1
            return [f"{actor} takes 1 coin (now {coins[actor]})."]
        if kind == "foreign_aid":
            coins[actor] += 2
            return [f"{actor} takes 2 coins (now {coins[actor]})."]
        if kind == "tax":
            coins[actor] += 3
            return [f"{actor} takes 3 coins (now {coins[actor]})."]
        if kind == "steal":
            amount = min(2, coins[target])
            coins[target] -= amount
            coins[actor] += amount
            return [f"{actor} takes {amount} coin{'s' if amount != 1 else ''} from {target}."]
        if kind == "exchange":
            ctx["drawn"] = [self.state["deck"].pop(), self.state["deck"].pop()]
            ctx["stage"] = "exchange"
            self.state["phase_counter"] += 1
            return [f"{actor} draws two cards from the deck."]
        # coup or assassinate: the target loses an influence, if it is still in the game
        if self.alive(player_id=target):
            self.state["pending_loss"].append(target)
            self.state["phase_counter"] += 1
            return [f"The {ACTION_NAMES[kind]} against {target} succeeds."]
        return [f"{target} is already out; the {ACTION_NAMES[kind]} has no effect."]

    def _end_turn(self) -> list[str]:
        """The next living player's turn starts, and the thinking setting decides who thinks."""
        self.state["turn"] += 1
        if self.turn_limit_reached:
            return [f"The turn limit ({self.setting(key='max_turns')}) is reached; nobody wins."]
        self.state["actor"] = self.next_living(after=self.state["actor"])
        self.state["phase_counter"] += 1
        # "actor" is option B in RULES.md (the next actor only); "all" is option A.
        thinkers = self.living() if self.setting(key="thinking") == "all" else [self.state["actor"]]
        self.request_thinking(player_ids=thinkers)
        return []

    def _log(self, *, events: list[str]) -> None:
        self.state["log"].extend(f"Turn {self.state['turn']}: {text}" for text in events)
