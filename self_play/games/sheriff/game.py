"""Sheriff of Nottingham, base game of the first edition. See RULES.md for the rules, their
sources and every interpretation made here.

3 to 5 players. Each round has one Sheriff (everyone else is a merchant) and four parts: the
Market, loading, declaring and inspection. The Sheriff passes left after each round; each
player is Sheriff 3 times with 3 players and 2 times with 4 or 5.

Each decision is its own phase, and every big choice is a chain of small ones, so that no
legal list is long:

    set_aside      ACTION        the market merchant  one card at a time, then "done"
    draw           ACTION        the market merchant  one card at a time, discard piles first
    discard_pile   ACTION        the market merchant  the pile its set-aside cards go on
    discard_order  ACTION        the market merchant  one card at a time; the last is the top
    load           SIMULTANEOUS  merchants loading    one card at a time, then "done"
    declare        ACTION        each merchant        one legal type; the game adds the count
    pick_bag       ACTION        the Sheriff          the next bag to deal with
    negotiate      DISCUSSION    merchant, Sheriff    talk, then the merchant's gold offer
    offer_goods    ACTION        the merchant         goods added one at a time, then "done"
    decide         ACTION        the Sheriff          "pass" (take the offer) or "inspect"
    confiscate_pile  ACTION      the Sheriff          the pile confiscated goods go on
    confiscate_order ACTION      the Sheriff          one card at a time; the last is the top
    pay            ACTION        a player short of gold  stand goods one at a time

In `negotiate` both talk, but only the merchant acts: its offer opens the next phase, so the
runner never asks the Sheriff for an action there.

`state["stage"]` is the part of the round being played: "market", "load", "declare" or
"inspect"; `state["inspection"]` is the bag being dealt with and its `step`, or None while
the Sheriff picks the next bag. A load phase stays open (same phase_id) until every merchant
still loading has answered for this card; every other action opens a new phase.

Thinking: everyone thinks once at the opening (every game's start) and again at the start
of every later round. game_config "thinking" = "deal" also asks the merchant and the Sheriff
before each negotiation.
"""

from collections import Counter
from typing import Any

from self_play.game import DialogueMode, DialoguePhase, Game
from self_play.games.sheriff.prompts import NAMES, answer_request, describe, parse_action, render_prompt, rules_text

GOODS = {"apples": (2, 2), "cheese": (3, 2), "bread": (3, 2), "chickens": (4, 2),
         "pepper": (6, 4), "mead": (7, 4), "silk": (8, 4), "crossbow": (9, 4)}
"""Every good, in a fixed order, with its (value, penalty) in gold."""
LEGAL = ("apples", "cheese", "bread", "chickens")
CONTRABAND = ("pepper", "mead", "silk", "crossbow")
COUNTS = {
    3: {"apples": 48, "cheese": 36, "bread": 0, "chickens": 24, "pepper": 18, "mead": 16, "silk": 9, "crossbow": 5},
    4: {"apples": 48, "cheese": 36, "bread": 36, "chickens": 24, "pepper": 22, "mead": 21, "silk": 12, "crossbow": 5},
}
"""The cards in play. The 3-player game leaves out the cards marked 4+; 5 players use the 4-player deck."""
BONUSES = {"apples": (20, 10), "cheese": (15, 10), "bread": (15, 10), "chickens": (10, 5)}
"""The (King, Queen) bonus of each legal good."""
SHERIFF_TURNS = {3: 3, 4: 2, 5: 2}
"""How many times each player is Sheriff, by player count."""
MIN_PLAYERS, MAX_PLAYERS = 3, 5
START_GOLD, HAND_SIZE, MAX_BAG, MAX_SET_ASIDE = 50, 6, 5, 5
DISCARD_START = 5
"""Cards turned face up onto each of the two discard piles at setup."""
RESHUFFLE_KEEP = 5
"""Cards left on top of each discard pile when the draw pile is rebuilt."""
THINKING = ("round", "deal")
DEFAULTS = {"talk_rounds": 2, "thinking": "round"}


def check_count(*, key: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, not {value!r}")


def deck_counts(*, players: int) -> dict[str, int]:
    return COUNTS[min(players, 4)]


def distinct(*, cards: list[str]) -> list[str]:
    """The goods among `cards`, each once, in GOODS order."""
    return [good for good in GOODS if good in cards]


def honest_order(*, cards: list[str]) -> list[str]:
    """The goods among `cards`, each once: legal goods first, the most numerous first (ties
    in GOODS order), then contraband in GOODS order. Its first good is the honest choice."""
    counts = Counter(cards)
    legal = sorted((good for good in LEGAL if counts[good]), key=lambda good: -counts[good])
    return legal + [good for good in CONTRABAND if counts[good]]


def sort_goods(*, cards: list[str]) -> list[str]:
    order = list(GOODS)
    return sorted(cards, key=order.index)


class SheriffGame(Game):
    """game_config: {"seed": int, "players": 3..5, "talk_rounds": int >= 0 (default 2, the
    negotiation), "thinking": "round" (default) or "deal"}."""

    name = "sheriff"

    def initial_state(self) -> dict[str, Any]:
        config = {**DEFAULTS, **self.game_config}
        players = config["players"]
        if not isinstance(players, int) or isinstance(players, bool) or not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be an integer from {MIN_PLAYERS} to {MAX_PLAYERS}, not {players!r}")
        check_count(key="talk_rounds", value=config["talk_rounds"], minimum=0)
        if config["thinking"] not in THINKING:
            raise ValueError(f"thinking must be one of {', '.join(THINKING)}, not {config['thinking']!r}")
        ids = [f"p{i}" for i in range(1, players + 1)]
        deck = [good for good, copies in deck_counts(players=players).items() for _ in range(copies)]
        self.rng(label="deal", index=0).shuffle(deck)
        hands = {pid: [deck.pop() for _ in range(HAND_SIZE)] for pid in ids}
        discards = [[deck.pop() for _ in range(DISCARD_START)] for _ in range(2)]
        # The rulebook starts with the player holding the most real cash; ties are random.
        sheriff = self.rng(label="sheriff", index=0).choice(ids)
        rounds = SHERIFF_TURNS[players] * players
        return {
            "players": ids,
            "hands": hands,            # hidden: each player sees only its own
            "deck": deck,              # hidden: only its size is public; the top card is the last
            "discards": discards,      # face up; the top card is the last; only tops and sizes are shown
            "reshuffles": 0,           # draw piles rebuilt so far, the index of the next shuffle
            "gold": {pid: START_GOLD for pid in ids},
            "stand_legal": {pid: [] for pid in ids},       # face up
            "stand_contraband": {pid: [] for pid in ids},  # face down: only the count is public
            "round": 1,
            "rounds": rounds,
            "sheriff": sheriff,
            "sheriffs": [sheriff],     # the Sheriff of each round so far
            "stage": "market",
            "market": self._new_market(merchants=self._clockwise(ids=ids, after=sheriff)),
            "loaders": [],             # merchants who load a bag this round
            "loading": [],             # those still loading at the start of this card
            "answered": [],            # those who answered for this card
            "finished": [],            # those whose bag is closed
            "bags": {pid: [] for pid in ids},              # hidden until opened
            "declarations": {pid: None for pid in ids},    # {"good", "count"} once declared
            "to_declare": [],
            "unresolved": [],          # bags the Sheriff has not dealt with, clockwise
            "inspection": None,        # the bag being dealt with, see `_new_inspection`
            "phase_counter": 0,
            "log": [f"The game begins. Round 1 of {rounds}: {sheriff} is the Sheriff."],
        }

    # --- helpers -------------------------------------------------------------------

    def setting(self, *, key: str) -> Any:
        return self.game_config.get(key, DEFAULTS[key])

    @staticmethod
    def _clockwise(*, ids: list[str], after: str) -> list[str]:
        """Every player except `after`, clockwise from its left."""
        start = ids.index(after)
        return [ids[(start + offset) % len(ids)] for offset in range(1, len(ids))]

    @staticmethod
    def _new_market(*, merchants: list[str]) -> dict[str, Any]:
        return {"merchants": merchants, "step": "set_aside", "set_aside": [], "pile": None, "deck_drawn": False}

    @staticmethod
    def _new_inspection(*, merchant: str) -> dict[str, Any]:
        """The bag being dealt with. Everything in it is public: the offer (gold and goods named),
        the confiscated goods and the debt still owed."""
        return {"merchant": merchant, "step": "negotiate", "offer": {"gold": 0, "stand": [], "bag": []},
                "confiscated": [], "pile": None, "debt": None}

    def merchants(self) -> list[str]:
        """This round's merchants, clockwise from the Sheriff's left."""
        return self._clockwise(ids=self.player_ids, after=self.state["sheriff"])

    @property
    def done(self) -> bool:
        return self.state["round"] > self.state["rounds"]

    def _deck_available(self) -> bool:
        """Whether a card can be drawn from the draw pile, rebuilding it if need be."""
        return bool(self.state["deck"]) or any(len(pile) > RESHUFFLE_KEEP for pile in self.state["discards"])

    def _draw_from_deck(self, *, events: list[str]) -> str | None:
        """The top card of the draw pile. An empty draw pile is first rebuilt from every
        discard pile except its top five cards. None if no card is left even then."""
        deck = self.state["deck"]
        if not deck:
            pool: list[str] = []
            for pile in self.state["discards"]:
                pool += pile[:-RESHUFFLE_KEEP]
                del pile[:-RESHUFFLE_KEEP]
            if pool:
                self.rng(label="reshuffle", index=self.state["reshuffles"]).shuffle(pool)
                self.state["reshuffles"] += 1
                deck.extend(pool)
                events.append(f"The draw pile is empty: {len(pool)} cards from under the top {RESHUFFLE_KEEP} "
                              f"of each discard pile are shuffled into a new draw pile.")
        return deck.pop() if deck else None

    # --- what is happening now -------------------------------------------------

    @property
    def player_ids(self) -> list[str]:
        return list(self.state["players"])

    def _get_dialogue_phase(self) -> DialoguePhase | None:
        if self.done:
            return None
        state = self.state
        phase_id, stage = state["phase_counter"], state["stage"]
        if stage == "market":
            market = state["market"]
            return DialoguePhase(phase_id=phase_id, phase=market["step"], mode=DialogueMode.ACTION,
                                 eligible=(market["merchants"][0],))
        if stage == "load":
            waiting = tuple(pid for pid in state["loading"] if pid not in state["answered"])
            return DialoguePhase(phase_id=phase_id, phase="load", mode=DialogueMode.SIMULTANEOUS, eligible=waiting)
        if stage == "declare":
            return DialoguePhase(phase_id=phase_id, phase="declare", mode=DialogueMode.ACTION,
                                 eligible=(state["to_declare"][0],))
        sheriff, inspection = state["sheriff"], state["inspection"]
        if inspection is None:
            return DialoguePhase(phase_id=phase_id, phase="pick_bag", mode=DialogueMode.ACTION, eligible=(sheriff,))
        step, merchant = inspection["step"], inspection["merchant"]
        if step == "negotiate":
            return DialoguePhase(phase_id=phase_id, phase="negotiate", mode=DialogueMode.DISCUSSION,
                                 eligible=(merchant, sheriff), talk_rounds=self.setting(key="talk_rounds"))
        actor = {"offer_goods": merchant, "pay": (inspection["debt"] or {}).get("payer")}.get(step, sheriff)
        return DialoguePhase(phase_id=phase_id, phase=step, mode=DialogueMode.ACTION, eligible=(actor,))

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        # The first listed action is the honest or do-nothing one: the runner's fallback.
        phase = self._get_dialogue_phase()
        if phase is None or player_id not in phase.eligible:
            return []
        state = self.state
        market, hand = state["market"], state["hands"][player_id]
        if phase.phase == "set_aside":
            if len(market["set_aside"]) >= MAX_SET_ASIDE:
                return ["done"]
            return ["done"] + [{"set_aside": good} for good in distinct(cards=hand)]
        if phase.phase == "draw":
            return self._draw_sources()
        if phase.phase == "discard_pile":
            return [{"pile": 1}, {"pile": 2}]
        if phase.phase == "discard_order":
            return [{"place": good} for good in distinct(cards=market["set_aside"])]
        if phase.phase == "load":
            loads = [{"load": good} for good in honest_order(cards=hand)]
            return ["done"] + loads if state["bags"][player_id] else loads
        if phase.phase == "declare":
            types = [good for good in LEGAL if deck_counts(players=len(self.player_ids))[good]]
            true = [good for good in honest_order(cards=state["bags"][player_id]) if good in LEGAL][:1]
            return [{"declare": good} for good in true + [good for good in types if good not in true]]
        if phase.phase == "pick_bag":
            return [{"bag": pid} for pid in state["unresolved"]]
        inspection = state["inspection"]
        if phase.phase == "negotiate":  # the Sheriff talks here but does not act
            return [{"gold": n} for n in range(state["gold"][player_id] + 1)] if player_id == inspection["merchant"] else []
        if phase.phase == "offer_goods":
            return ["done"] + [{"add": {"from": source, "good": good}} for source in ("stand", "bag")
                               for good in distinct(cards=self._offerable(source=source))]
        if phase.phase == "decide":
            return ["pass", "inspect"]
        if phase.phase == "confiscate_pile":
            return [{"pile": 1}, {"pile": 2}]
        if phase.phase == "confiscate_order":
            return [{"place": good} for good in distinct(cards=inspection["confiscated"])]
        return [{"pay": good} for good in self._payable(player_id=player_id)]

    def _offerable(self, *, source: str) -> list[str]:
        """The merchant's goods in its stand or bag not yet in the offer."""
        state, inspection = self.state, self.state["inspection"]
        merchant = inspection["merchant"]
        cards = (state["stand_legal"][merchant] + state["stand_contraband"][merchant] if source == "stand"
                 else state["bags"][merchant])
        return list((Counter(cards) - Counter(inspection["offer"][source])).elements())

    def _payable(self, *, player_id: str) -> list[str]:
        """The goods a player short of gold may give: legal goods while it has any, then
        contraband; each good once, the cheapest first (ties in GOODS order)."""
        stand = self.state["stand_legal"][player_id] or self.state["stand_contraband"][player_id]
        return sorted(distinct(cards=stand), key=lambda good: GOODS[good][0])

    def _draw_sources(self) -> list[dict[str, str]]:
        """The draw pile first; the discard piles with cards, until the draw pile is used."""
        market = self.state["market"]
        sources = [{"draw": "deck"}] if self._deck_available() else []
        if not market["deck_drawn"]:
            sources += [{"draw": f"discard_{i}"} for i, pile in enumerate(self.state["discards"], start=1) if pile]
        return sources

    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Public: gold, hand sizes, the number of set-aside cards, every stand's legal goods
        and its number of contraband, declarations, the bags still to be dealt with, the
        draw pile's size, each discard pile's top card and size, the bag being dealt with (its
        offer, with every offered good named, confiscated goods and the debt still owed), the
        round, the Sheriff, who is deciding and the log. Private: this player's hand, set-aside cards, bag and
        face-down contraband. Never another player's hand, set-aside cards, bag or face-down
        contraband, nor the draw pile's order. While bags are loaded, everyone loading is
        shown as deciding until all are done, so nobody learns when another bag closed."""
        state = self.state
        phase = self._get_dialogue_phase()
        market = state["market"]
        marketeer = market["merchants"][0] if state["stage"] == "market" and not self.done else None
        deciding = list(state["loaders"]) if phase is not None and phase.phase == "load" else (
            list(phase.eligible) if phase else [])
        return {
            "you": player_id,
            "your_hand": sort_goods(cards=state["hands"][player_id]),
            "your_set_aside": sort_goods(cards=market["set_aside"]) if marketeer == player_id else [],
            "your_bag": sort_goods(cards=state["bags"][player_id]),
            "your_contraband": sort_goods(cards=state["stand_contraband"][player_id]),
            "players": [
                {
                    "id": pid,
                    "gold": state["gold"][pid],
                    "hand_size": len(state["hands"][pid]),
                    "set_aside": len(market["set_aside"]) if marketeer == pid else 0,
                    "stand": dict(Counter(sort_goods(cards=state["stand_legal"][pid]))),
                    "contraband": len(state["stand_contraband"][pid]),
                    "declared": dict(state["declarations"][pid]) if state["declarations"][pid] else None,
                }
                for pid in self.player_ids
            ],
            "round": state["round"],
            "rounds": state["rounds"],
            "sheriff": state["sheriff"],
            "draw_pile": len(state["deck"]),
            "discard_piles": [{"top": pile[-1] if pile else None, "size": len(pile)} for pile in state["discards"]],
            "unresolved": list(state["unresolved"]) if state["stage"] == "inspect" else [],
            "inspection": self._public_inspection(),
            "phase": phase.phase if phase else None,
            "deciding": deciding,
            "log": list(state["log"]),
            "winners": self.get_outcome()["winners"],
        }

    def _public_inspection(self) -> dict[str, Any] | None:
        inspection = self.state["inspection"]
        if inspection is None or self.done:
            return None
        offer = inspection["offer"]
        return {
            "merchant": inspection["merchant"],
            "offer": {"gold": offer["gold"], "stand": sort_goods(cards=offer["stand"]), "bag": sort_goods(cards=offer["bag"])},
            "confiscated": sort_goods(cards=inspection["confiscated"]),
            "debt": dict(inspection["debt"]) if inspection["debt"] else None,
        }

    # --- scoring -----------------------------------------------------------------

    def scores(self) -> dict[str, dict[str, int]]:
        """Each player's goods (the value of its whole stand), gold, King and Queen bonuses,
        and total. For each legal good, the most cards in a stand is King and the next most
        is Queen; a player needs at least one card of the good for either."""
        state, ids = self.state, self.player_ids
        bonus = {pid: 0 for pid in ids}
        for good, (king, queen) in BONUSES.items():
            counts = {pid: state["stand_legal"][pid].count(good) for pid in ids}
            top = max(counts.values())
            if top == 0:
                continue
            kings = [pid for pid in ids if counts[pid] == top]
            if len(kings) > 1:  # tied Kings share King plus Queen, rounded down; no Queen is paid
                for pid in kings:
                    bonus[pid] += (king + queen) // len(kings)
                continue
            bonus[kings[0]] += king
            second = max((counts[pid] for pid in ids if pid not in kings), default=0)
            if second:
                queens = [pid for pid in ids if pid not in kings and counts[pid] == second]
                for pid in queens:
                    bonus[pid] += queen // len(queens)
        table = {}
        for pid in ids:
            goods = sum(GOODS[card][0] for card in state["stand_legal"][pid] + state["stand_contraband"][pid])
            gold = state["gold"][pid]
            table[pid] = {"goods": goods, "gold": gold, "bonus": bonus[pid], "total": goods + gold + bonus[pid]}
        return table

    def winners(self) -> list[str]:
        """The highest total; ties go to the most legal goods, then the most contraband; a tie
        after that is shared."""
        table = self.scores()

        def rank(pid: str) -> tuple[int, int, int]:
            return (table[pid]["total"], len(self.state["stand_legal"][pid]), len(self.state["stand_contraband"][pid]))

        best = max(rank(pid) for pid in self.player_ids)
        return [pid for pid in self.player_ids if rank(pid) == best]

    def get_outcome(self) -> dict[str, Any]:
        """The winner gets 1; tied winners share it. Scores are added once the game is over."""
        if not self.done:
            return {"done": False, "rewards": {pid: 0.0 for pid in self.player_ids}, "winners": [], "scores": None,
                    "termination_reason": None}
        winners = self.winners()
        rewards = {pid: 1.0 / len(winners) if pid in winners else 0.0 for pid in self.player_ids}
        return {"done": True, "rewards": rewards, "winners": winners, "scores": self.scores(),
                "termination_reason": "all_rounds_played"}

    # --- text for players --------------------------------------------------------

    def get_rules(self, *, observation: dict[str, Any]) -> str:
        players = len(observation["players"])
        return rules_text(players=players, sheriff_turns=SHERIFF_TURNS[players], rounds=observation["rounds"],
                          bread=bool(deck_counts(players=players)["bread"]))

    def render_observation(self, *, observation: dict[str, Any]) -> str:
        return render_prompt(rules=self.get_rules(observation=observation), observation=observation)

    def render_answer_request(self, *, legal_actions: list[Any]) -> str:
        return answer_request(legal_actions=legal_actions)

    def parse_answer(self, *, player_id: str, answer: str) -> Any:
        return parse_action(answer=answer)

    # --- transitions -------------------------------------------------------------

    def _apply(self, *, player_id: str, action: Any) -> list[str]:
        phase = self._get_dialogue_phase().phase
        round_number = self.state["round"]
        events: list[str] = []
        if phase == "set_aside":
            self._set_aside(player_id=player_id, action=action, events=events)
        elif phase == "draw":
            self._draw(player_id=player_id, source=action["draw"], events=events)
        elif phase in ("discard_pile", "discard_order"):
            self._market_discard(player_id=player_id, action=action, events=events)
        elif phase == "load":
            self._load(player_id=player_id, action=action, events=events)
        elif phase == "declare":
            self._declare(player_id=player_id, good=action["declare"], events=events)
        elif phase == "pick_bag":
            self._pick_bag(merchant=action["bag"], events=events)
        elif phase == "negotiate":
            self._offer_gold(player_id=player_id, gold=action["gold"], events=events)
        elif phase == "offer_goods":
            self._offer_goods(player_id=player_id, action=action, events=events)
        elif phase == "decide":
            self._decide(action=action, events=events)
        elif phase in ("confiscate_pile", "confiscate_order"):
            self._confiscate(player_id=player_id, action=action, events=events)
        else:
            self._pay_good(good=action["pay"], events=events)
        # A load phase stays open until everyone still loading has answered for this card.
        if not (self.state["stage"] == "load" and self.state["answered"]):
            self.state["phase_counter"] += 1
        self.state["log"].extend(f"Round {round_number}: {text}" for text in events)
        return events

    # The Market

    def _set_aside(self, *, player_id: str, action: Any, events: list[str]) -> None:
        market = self.state["market"]
        if action != "done":
            self.state["hands"][player_id].remove(action["set_aside"])
            market["set_aside"].append(action["set_aside"])
            return
        count = len(market["set_aside"])
        if not count:
            events.append(f"{player_id} keeps their hand in the Market.")
            self._next_market(events=events)
            return
        events.append(f"{player_id} sets aside {count} card{'s' if count != 1 else ''} face down.")
        market["step"] = "draw"
        self._end_draws_if_full(player_id=player_id)

    def _draw(self, *, player_id: str, source: str, events: list[str]) -> None:
        market, hand = self.state["market"], self.state["hands"][player_id]
        if source == "deck":
            hand.append(self._draw_from_deck(events=events))
            market["deck_drawn"] = True
            events.append(f"{player_id} draws a card from the draw pile.")
        else:
            index = int(source.split("_")[1])
            card = self.state["discards"][index - 1].pop()
            hand.append(card)
            events.append(f"{player_id} takes {NAMES[card]} from discard pile {index}.")
        self._end_draws_if_full(player_id=player_id)

    def _end_draws_if_full(self, *, player_id: str) -> None:
        """Draws end at 6 cards in hand, or when there is nothing left to draw."""
        if len(self.state["hands"][player_id]) >= HAND_SIZE or not self._draw_sources():
            self.state["market"]["step"] = "discard_pile"

    def _market_discard(self, *, player_id: str, action: Any, events: list[str]) -> None:
        market = self.state["market"]
        if "pile" in action:
            market["pile"], market["step"] = action["pile"], "discard_order"
        if self._place_cards(player_id=player_id, cards=market["set_aside"], pile=market["pile"],
                             good=action.get("place"), events=events):
            self._next_market(events=events)

    def _place_cards(self, *, player_id: str, cards: list[str], pile: int, good: str | None, events: list[str]) -> bool:
        """Put `good` (if given) from `cards` on top of the discard pile; then, when only one good
        is left, its order is no choice, so place it all. True once `cards` is empty."""
        placing = [good] if good else []
        rest = list((Counter(cards) - Counter(placing)).elements())
        if len(distinct(cards=rest)) == 1:
            placing += rest
        for card in placing:
            cards.remove(card)
            self.state["discards"][pile - 1].append(card)
            events.append(f"{player_id} puts {NAMES[card]} on discard pile {pile}.")
        return not cards

    def _next_market(self, *, events: list[str]) -> None:
        rest = self.state["market"]["merchants"][1:]
        if rest:
            self.state["market"] = self._new_market(merchants=rest)
            return
        self.state["market"] = self._new_market(merchants=[])
        self._start_loading(events=events)

    # Loading and declaring

    def _start_loading(self, *, events: list[str]) -> None:
        state = self.state
        loaders = [pid for pid in self.merchants() if state["hands"][pid]]
        state.update(stage="load", loaders=loaders, loading=list(loaders), answered=[], finished=[])
        if not loaders:
            self._start_declaring(events=events)

    def _load(self, *, player_id: str, action: Any, events: list[str]) -> None:
        state = self.state
        hand, bag = state["hands"][player_id], state["bags"][player_id]
        state["answered"].append(player_id)
        if action == "done":
            state["finished"].append(player_id)
        else:
            hand.remove(action["load"])
            bag.append(action["load"])
            if len(bag) >= MAX_BAG or not hand:
                state["finished"].append(player_id)
        if any(pid not in state["answered"] for pid in state["loading"]):
            return  # others still owe their answer for this card
        state["loading"] = [pid for pid in state["loading"] if pid not in state["finished"]]
        state["answered"] = []
        if not state["loading"]:
            events.append("Every bag is loaded and closed.")
            self._start_declaring(events=events)

    def _start_declaring(self, *, events: list[str]) -> None:
        self.state["stage"] = "declare"
        self.state["to_declare"] = [pid for pid in self.merchants() if self.state["bags"][pid]]
        if not self.state["to_declare"]:
            self._start_inspection(events=events)

    def _declare(self, *, player_id: str, good: str, events: list[str]) -> None:
        count = len(self.state["bags"][player_id])
        self.state["declarations"][player_id] = {"good": good, "count": count}
        self.state["to_declare"].pop(0)
        events.append(f"{player_id} declares {count} {NAMES[good]}.")
        if not self.state["to_declare"]:
            self._start_inspection(events=events)

    # Inspection

    def _start_inspection(self, *, events: list[str]) -> None:
        self.state["stage"] = "inspect"
        self.state["unresolved"] = [pid for pid in self.merchants() if self.state["bags"][pid]]
        if not self.state["unresolved"]:
            self._end_round(events=events)

    def _pick_bag(self, *, merchant: str, events: list[str]) -> None:
        self.state["unresolved"].remove(merchant)
        self.state["inspection"] = self._new_inspection(merchant=merchant)
        events.append(f"The Sheriff {self.state['sheriff']} takes up {merchant}'s bag.")
        if self.setting(key="thinking") == "deal":
            self.request_thinking(player_ids=[merchant, self.state["sheriff"]])

    def _offer_gold(self, *, player_id: str, gold: int, events: list[str]) -> None:
        self.state["inspection"]["offer"]["gold"] = gold
        self.state["inspection"]["step"] = "offer_goods"
        events.append(f"{player_id} offers {gold} gold.")

    def _offer_goods(self, *, player_id: str, action: Any, events: list[str]) -> None:
        inspection = self.state["inspection"]
        if action == "done":
            inspection["step"] = "decide"
            events.append(f"{player_id}'s offer: {self._describe_offer()}.")
            return
        source, good = action["add"]["from"], action["add"]["good"]
        inspection["offer"][source].append(good)
        events.append(f"{player_id} adds {NAMES[good]} from their {source} to the offer.")

    def _describe_offer(self) -> str:
        offer = self.state["inspection"]["offer"]
        parts = [f"{offer['gold']} gold"]
        parts += [f"{describe(cards=offer[source])} from the {source}" for source in ("stand", "bag") if offer[source]]
        return ", ".join(parts)

    def _decide(self, *, action: str, events: list[str]) -> None:
        state, inspection = self.state, self.state["inspection"]
        sheriff, merchant = state["sheriff"], inspection["merchant"]
        if action == "pass":
            offer = inspection["offer"]
            state["gold"][merchant] -= offer["gold"]
            state["gold"][sheriff] += offer["gold"]
            for good in offer["stand"]:
                stand = state["stand_legal" if good in LEGAL else "stand_contraband"][merchant]
                stand.remove(good)
                self._to_stand(player_id=sheriff, cards=[good])
            for good in offer["bag"]:
                state["bags"][merchant].remove(good)
            self._to_stand(player_id=sheriff, cards=offer["bag"])
            events.append(f"{sheriff} takes {merchant}'s offer ({self._describe_offer()}) and lets the bag pass.")
            self._pass_bag(merchant=merchant, events=events)
            self._finish_bag(events=events)
            return
        bag, declared = state["bags"][merchant], state["declarations"][merchant]["good"]
        events.append(f"{sheriff} refuses the offer and inspects {merchant}'s bag: {describe(cards=bag)}.")
        truthful = [card for card in bag if card == declared]
        others = [card for card in bag if card != declared]
        self._to_stand(player_id=merchant, cards=truthful)
        state["bags"][merchant] = []
        if not others:
            amount = GOODS[declared][1] * len(truthful)
            events.append(f"The bag is honest: {sheriff} owes {merchant} {amount} gold in penalties.")
            inspection["debt"] = {"payer": sheriff, "payee": merchant, "remaining": amount}
            self._pay_gold(events=events)
            return
        amount = sum(GOODS[card][1] for card in others)
        events.append(f"{merchant} lied: {describe(cards=truthful)} go to the stand, {describe(cards=others)} "
                      f"are confiscated, and {merchant} owes {sheriff} {amount} gold in penalties.")
        inspection["confiscated"] = others
        inspection["debt"] = {"payer": merchant, "payee": sheriff, "remaining": amount}
        inspection["step"] = "confiscate_pile"

    def _confiscate(self, *, player_id: str, action: Any, events: list[str]) -> None:
        inspection = self.state["inspection"]
        if "pile" in action:
            inspection["pile"], inspection["step"] = action["pile"], "confiscate_order"
        if self._place_cards(player_id=player_id, cards=inspection["confiscated"], pile=inspection["pile"],
                             good=action.get("place"), events=events):
            self._pay_gold(events=events)

    def _pay_gold(self, *, events: list[str]) -> None:
        """Pay the debt in gold, as far as it goes; the rest is paid in goods from the stand."""
        state, debt = self.state, self.state["inspection"]["debt"]
        payer, payee = debt["payer"], debt["payee"]
        paid = min(state["gold"][payer], debt["remaining"])
        state["gold"][payer] -= paid
        state["gold"][payee] += paid
        debt["remaining"] -= paid
        events.append(f"{payer} pays {payee} {paid} gold.")
        if debt["remaining"]:
            events.append(f"{payer} is {debt['remaining']} gold short and pays with goods from their stand.")
        self._settle(events=events)

    def _pay_good(self, *, good: str, events: list[str]) -> None:
        state, debt = self.state, self.state["inspection"]["debt"]
        payer, payee = debt["payer"], debt["payee"]
        state["stand_legal" if good in LEGAL else "stand_contraband"][payer].remove(good)
        self._to_stand(player_id=payee, cards=[good])
        debt["remaining"] = max(0, debt["remaining"] - GOODS[good][0])
        events.append(f"{payer} gives {payee} {NAMES[good]} (worth {GOODS[good][0]}) from their stand.")
        self._settle(events=events)

    def _settle(self, *, events: list[str]) -> None:
        """The debt is paid once covered (no change is given) or once the payer's stand is empty
        (the rest is forgiven); until then the payer gives another good."""
        state, debt = self.state, self.state["inspection"]["debt"]
        if debt["remaining"] and self._payable(player_id=debt["payer"]):
            state["inspection"]["step"] = "pay"
            return
        if debt["remaining"]:
            events.append(f"{debt['payer']}'s stand is empty: the remaining {debt['remaining']} gold is forgiven.")
        self._finish_bag(events=events)

    def _to_stand(self, *, player_id: str, cards: list[str]) -> None:
        self.state["stand_legal"][player_id] += [card for card in cards if card in LEGAL]
        self.state["stand_contraband"][player_id] += [card for card in cards if card in CONTRABAND]

    def _pass_bag(self, *, merchant: str, events: list[str]) -> None:
        """What is left in a bag that passes: legal goods face up, contraband face down."""
        state = self.state
        bag = state["bags"][merchant]
        legal = [card for card in bag if card in LEGAL]
        self._to_stand(player_id=merchant, cards=bag)
        state["bags"][merchant] = []
        events.append(f"{describe(cards=legal)} face up and {len(bag) - len(legal)} contraband face down "
                      f"go to {merchant}'s stand.")

    def _finish_bag(self, *, events: list[str]) -> None:
        self.state["inspection"] = None
        if not self.state["unresolved"]:
            self._end_round(events=events)

    # The end of a round

    def _end_round(self, *, events: list[str]) -> None:
        state = self.state
        state["declarations"] = {pid: None for pid in self.player_ids}
        state["unresolved"] = []
        last = state["round"] == state["rounds"]
        if not last:  # everyone draws back to 6, merchants clockwise from the Sheriff's left, then the Sheriff
            for pid in self.merchants() + [state["sheriff"]]:
                drawn = 0
                while len(state["hands"][pid]) < HAND_SIZE and (card := self._draw_from_deck(events=events)):
                    state["hands"][pid].append(card)
                    drawn += 1
                if drawn:
                    events.append(f"{pid} draws {drawn} card{'s' if drawn != 1 else ''} from the draw pile.")
        state["round"] += 1
        if last:
            table = self.scores()
            events.append("The last round is over. Final scores: " + "; ".join(
                f"{pid} {row['total']} (goods {row['goods']}, gold {row['gold']}, bonuses {row['bonus']})"
                for pid, row in table.items()) + ".")
            events.append(f"Winner{'s' if len(self.winners()) > 1 else ''}: {', '.join(self.winners())}.")
            return
        ids = self.player_ids
        state["sheriff"] = ids[(ids.index(state["sheriff"]) + 1) % len(ids)]
        state["sheriffs"].append(state["sheriff"])
        state["stage"] = "market"
        state["market"] = self._new_market(merchants=self.merchants())
        events.append(f"Round {state['round']} of {state['rounds']} begins: {state['sheriff']} is the Sheriff.")
        self.request_thinking(player_ids=ids)
