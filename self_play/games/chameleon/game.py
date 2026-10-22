"""The Chameleon, base game, one round. See RULES.md for the rules, their sources and every choice made here.

3 to 6 players. Phases, in the order they happen (a phase that cannot happen is skipped):

    thinking   THINKING      everyone                          0
    clues      DISCUSSION    everyone, dealer first, clockwise  1, talk required
    debate     DISCUSSION    everyone, dealer first, clockwise  talk_rounds (skipped at 0)
    thinking   THINKING      everyone
    vote       SIMULTANEOUS  everyone                          0
    tiebreak   ACTION        the dealer                        0   (only on a tie)
    guess      ACTION        the Chameleon                     0   (only if the accused is the Chameleon;
                                                                    twice with 3 players if the first is wrong)

The clue words and the debate are table talk. The game only sees a forced "done" from each speaker, which
closes the phase once everyone has spoken. Votes are kept in `state["votes"]` but shown to nobody until all are in.
"""

from typing import Any

from self_play.game import THINK, DialogueMode, DialoguePhase, Game
from self_play.games.chameleon.prompts import answer_request, parse_action, render_prompt, rules_text

MIN_PLAYERS, MAX_PLAYERS = 3, 6
DEFAULTS = {"talk_rounds": 1}
DONE = "done"

TOPICS = {
    "Animals": ["dog", "cat", "horse", "cow", "pig", "sheep", "chicken", "duck",
                "rabbit", "mouse", "lion", "bear", "wolf", "fox", "deer", "owl"],
    "Food": ["bread", "cheese", "rice", "pasta", "pizza", "soup", "salad", "egg",
             "butter", "honey", "cake", "apple", "potato", "tomato", "sausage", "noodle"],
    "Clothes": ["shirt", "jacket", "coat", "scarf", "glove", "hat", "sock", "shoe",
                "boot", "belt", "dress", "skirt", "jeans", "sweater", "tie", "sandal"],
    "Kitchen": ["spoon", "fork", "knife", "plate", "bowl", "cup", "pan", "pot",
                "oven", "fridge", "kettle", "toaster", "sink", "whisk", "tray", "jug"],
    "Around the house": ["bed", "chair", "table", "sofa", "lamp", "door", "window", "mirror",
                         "clock", "carpet", "shelf", "curtain", "stairs", "pillow", "desk", "drawer"],
    "Jobs": ["doctor", "nurse", "teacher", "farmer", "baker", "pilot", "driver", "chef",
             "plumber", "artist", "singer", "judge", "soldier", "writer", "dentist", "builder"],
    "Sports": ["soccer", "tennis", "golf", "boxing", "rugby", "hockey", "cricket", "baseball",
               "basketball", "swimming", "cycling", "skiing", "running", "surfing", "rowing", "archery"],
    "Transport": ["car", "bus", "train", "bicycle", "plane", "boat", "ship", "taxi",
                  "truck", "tram", "subway", "scooter", "helicopter", "motorcycle", "ferry", "rocket"],
    "Places in town": ["school", "hospital", "library", "bank", "museum", "cinema", "market", "bakery",
                       "church", "station", "park", "airport", "hotel", "factory", "stadium", "bridge"],
    "Nature": ["rain", "snow", "wind", "cloud", "storm", "fog", "thunder", "rainbow",
               "river", "mountain", "forest", "ocean", "desert", "island", "lake", "valley"],
}
"""Our own topic cards: 16 everyday words each, in the order a card lists them."""
TOPIC_NAMES = list(TOPICS)

REWARDS = {  # result -> (Chameleon, every other player): the rulebook's optional scores halved
    "escaped_undetected": (1.0, 0.0),
    "escaped_guess": (0.5, 0.0),
    "caught": (0.0, 1.0),
}


class ChameleonGame(Game):
    """game_config: {"seed": int, "players": 3..6, "talk_rounds": int >= 0 (default 1, the debate)}."""

    name = "chameleon"

    def initial_state(self) -> dict[str, Any]:
        config = {**DEFAULTS, **self.game_config}
        players = config["players"]
        if not isinstance(players, int) or isinstance(players, bool) or not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"players must be an integer from {MIN_PLAYERS} to {MAX_PLAYERS}, not {players!r}")
        rounds = config["talk_rounds"]
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
            raise ValueError(f"talk_rounds must be an integer >= 0, not {rounds!r}")
        ids = [f"p{i}" for i in range(1, players + 1)]
        dealer = self.rng(label="dealer", index=0).choice(ids)
        chameleon = self.rng(label="chameleon", index=0).choice(ids)
        topic = self.rng(label="topic", index=0).choice(TOPIC_NAMES)
        secret = self.rng(label="word", index=0).choice(TOPICS[topic])
        order = ids[ids.index(dealer):] + ids[:ids.index(dealer)]
        return {
            "players": ids,
            "dealer": dealer,
            "chameleon": chameleon,       # hidden from everyone else until the game ends
            "topic": topic,
            "words": list(TOPICS[topic]),
            "secret_word": secret,        # hidden from the Chameleon until the game ends
            "clue_order": order,          # the dealer first, clockwise
            "stage": "clues",             # clues, debate, vote, tiebreak, guess or over
            "remaining": list(order),     # who still has to close the open phase
            "votes": {},                  # voter -> target; shown to nobody until all are in
            "tied": [],
            "accused": None,
            "guesses_left": 0,
            "wrong_guesses": [],
            "result": None,               # escaped_undetected, escaped_guess or caught
            "phase_counter": 0,
            "log": [f"The game begins. {dealer} is the dealer and gives the first clue."],
        }

    # --- helpers -------------------------------------------------------------------

    def setting(self, *, key: str) -> Any:
        return self.game_config.get(key, DEFAULTS[key])

    @property
    def done(self) -> bool:
        return self.state["stage"] == "over"

    def _open_phase(self, *, stage: str, remaining: list[str]) -> None:
        self.state["stage"] = stage
        self.state["remaining"] = remaining
        self.state["phase_counter"] += 1

    # --- what is happening now -------------------------------------------------

    @property
    def player_ids(self) -> list[str]:
        return list(self.state["players"])

    def _get_dialogue_phase(self) -> DialoguePhase | None:
        state = self.state
        phase_id, stage = state["phase_counter"], state["stage"]
        if stage == "over":
            return None
        if stage == "clues":
            return DialoguePhase(phase_id=phase_id, phase="clues", mode=DialogueMode.DISCUSSION,
                                 eligible=tuple(state["remaining"]), talk_rounds=1, talk_required=True)
        if stage == "debate":
            return DialoguePhase(phase_id=phase_id, phase="debate", mode=DialogueMode.DISCUSSION,
                                 eligible=tuple(state["remaining"]), talk_rounds=self.setting(key="talk_rounds"))
        if stage == "vote":
            return DialoguePhase(phase_id=phase_id, phase="vote", mode=DialogueMode.SIMULTANEOUS,
                                 eligible=tuple(state["remaining"]))
        if stage == "tiebreak":
            return DialoguePhase(phase_id=phase_id, phase="tiebreak", mode=DialogueMode.ACTION,
                                 eligible=(state["dealer"],))
        return DialoguePhase(phase_id=phase_id, phase="guess", mode=DialogueMode.ACTION,
                             eligible=(state["chameleon"],))

    def _get_legal_actions(self, *, player_id: str) -> list[Any]:
        # The first listed action is the runner's fallback for an invalid answer.
        phase = self._get_dialogue_phase()
        if phase is None or player_id not in phase.eligible:
            return []
        if phase.phase in ("clues", "debate"):
            return [DONE]
        if phase.phase == "vote":
            return [{"vote": pid} for pid in self.player_ids if pid != player_id]
        if phase.phase == "tiebreak":
            return [{"choose": pid} for pid in self.state["tied"]]
        return [{"guess": word} for word in self.state["words"] if word not in self.state["wrong_guesses"]]

    def get_observation(self, *, player_id: str) -> dict[str, Any]:
        """Public: the players, the dealer, the topic and its words, the clue order, the phase, the log, and
        (once all are in) the votes, the tie and the accused. Private: whether you are the Chameleon and, if
        you are not, the secret word. Never the Chameleon's identity to anyone else, the secret word to the
        Chameleon, or anyone's vote before all are in. At the end everything is shown."""
        state = self.state
        phase = self._get_dialogue_phase()
        is_chameleon = player_id == state["chameleon"]
        votes_in = state["stage"] not in ("clues", "debate", "vote")
        return {
            "you": player_id,
            "is_chameleon": is_chameleon,
            "secret_word": None if is_chameleon and not self.done else state["secret_word"],
            "players": self.player_ids,
            "dealer": state["dealer"],
            "topic": state["topic"],
            "words": list(state["words"]),
            "clue_order": list(state["clue_order"]),
            "phase": phase.phase if phase else None,
            # While votes are coming in, nobody learns who has already voted: everyone is still "deciding".
            "deciding": (self.player_ids if phase.phase == "vote" else list(phase.eligible)) if phase else [],
            "votes": [[voter, state["votes"][voter]] for voter in self.player_ids if voter in state["votes"]] if votes_in else None,
            "tied": list(state["tied"]),
            "accused": state["accused"],
            "guesses_left": state["guesses_left"],
            "wrong_guesses": list(state["wrong_guesses"]),
            "log": list(state["log"]),
            "result": state["result"],
            "chameleon": state["chameleon"] if self.done else None,
        }

    def _final_rewards(self) -> dict[str, float]:
        result = self.state["result"]
        chameleon_reward, other_reward = REWARDS[result] if result else (0.0, 0.0)
        return {pid: chameleon_reward if pid == self.state["chameleon"] else other_reward for pid in self.player_ids}

    def get_outcome(self) -> dict[str, Any]:
        """The rulebook's optional scores, halved: see RULES.md."""
        state = self.state
        return {"done": self.done, "rewards": self._final_rewards(), "result": state["result"],
                "chameleon": state["chameleon"] if self.done else None,
                "secret_word": state["secret_word"] if self.done else None,
                "accused": state["accused"], "termination_reason": state["result"]}

    def get_step_rewards(self) -> list[float]:
        """A non-Chameleon's vote or tie-break earns 1 if it names the Chameleon; a guess earns 1 if
        right; thinking earns 0; every other step earns its player's final reward."""
        if not self.done:
            raise RuntimeError("step rewards are decided only once the game is over")
        chameleon, final = self.state["chameleon"], self._final_rewards()
        rewards = []
        for step in self.steps:
            player_id, action = step["player_id"], step["action"]
            if action == THINK:
                rewards.append(0.0)
            elif isinstance(action, dict) and "guess" in action:
                rewards.append(1.0 if action["guess"] == self.state["secret_word"] else 0.0)
            elif isinstance(action, dict) and player_id != chameleon and ("vote" in action or "choose" in action):
                rewards.append(1.0 if (action.get("vote") or action.get("choose")) == chameleon else 0.0)
            else:
                rewards.append(final[player_id])
        return rewards

    # --- text for players --------------------------------------------------------

    def get_rules(self, *, observation: dict[str, Any]) -> str:
        return rules_text(players=len(observation["players"]))

    def render_observation(self, *, observation: dict[str, Any]) -> str:
        return render_prompt(rules=self.get_rules(observation=observation), observation=observation)

    def render_answer_request(self, *, legal_actions: list[Any]) -> str:
        return answer_request(legal_actions=legal_actions)

    def parse_answer(self, *, player_id: str, answer: str) -> Any:
        return parse_action(answer=answer)

    # --- transitions -------------------------------------------------------------

    def _apply(self, *, player_id: str, action: Any) -> list[str]:
        state = self.state
        events: list[str] = []
        if action == DONE:
            state["remaining"].remove(player_id)
            if not state["remaining"]:
                events += self._after_talk()
        elif "vote" in action:
            state["votes"][player_id] = action["vote"]
            state["remaining"].remove(player_id)
            if not state["remaining"]:
                events += self._count_votes()
        elif "choose" in action:
            events += self._accuse(accused=action["choose"])
        else:
            events += self._guess(word=action["guess"])
        state["log"] += events
        return events

    def _after_talk(self) -> list[str]:
        """The clues or the debate are over: on to the next phase."""
        state = self.state
        after_clues = state["stage"] == "clues"
        if after_clues and self.setting(key="talk_rounds") > 0:
            self._open_phase(stage="debate", remaining=list(state["clue_order"]))
            return ["The clues are in. The debate begins."]
        self._open_phase(stage="vote", remaining=self.player_ids)
        self.request_thinking(player_ids=self.player_ids)
        return ["The clues are in. Everyone votes." if after_clues else "The debate is over. Everyone votes."]

    def _count_votes(self) -> list[str]:
        state = self.state
        votes = [[voter, state["votes"][voter]] for voter in self.player_ids]
        events = ["Votes: " + ", ".join(f"{voter} voted for {target}" for voter, target in votes) + "."]
        counts = {pid: sum(1 for _, target in votes if target == pid) for pid in self.player_ids}
        top = max(counts.values())
        tied = [pid for pid in self.player_ids if counts[pid] == top]
        if len(tied) == 1:
            return events + self._accuse(accused=tied[0])
        state["tied"] = tied
        self._open_phase(stage="tiebreak", remaining=[state["dealer"]])
        return events + [f"{', '.join(tied)} are tied with {top} votes each. The dealer, {state['dealer']}, chooses who is accused."]

    def _accuse(self, *, accused: str) -> list[str]:
        state = self.state
        state["accused"] = accused
        events = [f"{accused} is accused."]
        if accused != state["chameleon"]:
            events += [f"{accused} is not the Chameleon. The Chameleon, {state['chameleon']}, escapes undetected. "
                       f"The secret word was {state['secret_word']}."]
            state["result"] = "escaped_undetected"
            self._open_phase(stage="over", remaining=[])
            return events
        events.append(f"{accused} is the Chameleon and now guesses the secret word.")
        state["guesses_left"] = 2 if len(self.player_ids) == 3 else 1
        self._open_phase(stage="guess", remaining=[accused])
        return events

    def _guess(self, *, word: str) -> list[str]:
        state = self.state
        chameleon = state["chameleon"]
        if word == state["secret_word"]:
            state["result"] = "escaped_guess"
            self._open_phase(stage="over", remaining=[])
            return [f"{chameleon} guesses {word}. That is the secret word: the Chameleon escapes."]
        state["wrong_guesses"].append(word)
        state["guesses_left"] -= 1
        if state["guesses_left"] == 0:
            state["result"] = "caught"
            self._open_phase(stage="over", remaining=[])
            return [f"{chameleon} guesses {word}. That is wrong: the Chameleon is caught. "
                    f"The secret word was {state['secret_word']}."]
        self._open_phase(stage="guess", remaining=[chameleon])
        return [f"{chameleon} guesses {word}. That is wrong. One guess is left."]
