# The Chameleon

Base game of The Chameleon (Big Potato), one round per game, no scoring across rounds. 3 to 6 players. The base game is 4 to 6; the 3-player variant is included. The 7 and 8 player variant (the topic card is turned face down after the clues) is not supported yet.

Every rule below is marked **[rulebook]** (stated in the source) or **[our choice]** (needed to make the game playable by models, or not stated).

## Sources

- Rulebook, transcribed by RulesPal: https://rulespal.com/the-chameleon/rulebook (every **[rulebook]** rule below)

## Rules

**Setup**
- One player is the Chameleon; everyone else knows the secret word. **[rulebook]**
- A topic card shows a topic and 16 words; the secret word is one of them. Everyone, the Chameleon included, sees the topic card. **[rulebook]** (the dice and Code Cards that pick the word are replaced, see the next bullet)
- The seed picks the dealer, the Chameleon, the topic card and the secret word. This replaces the dice, the Code Cards and the Chameleon card. The dealer may be the Chameleon. **[our choice]**
- Topic cards are our own lists: at least 8 topics, 16 everyday words each. The printed cards are not copied. **[our choice]**
- Everyone except the Chameleon is told the secret word. The Chameleon is told that it is the Chameleon. **[rulebook]**
- Players sit in seat order `p1` … `pN`; "clockwise" is that order, wrapping round. **[our choice]**

**Clue round**
- Starting with the dealer and going clockwise, every player, the Chameleon included, says one word related to the secret word. **[rulebook]**
- Clues are public: every later player sees them. **[rulebook]**
- Repeating a word that was already said is allowed. **[rulebook]**
- A clue is a required talk turn: the engine rejects saying nothing (an empty answer or SILENCE) and asks again, and a player that still says nothing posts no clue and counts one invalid answer. **[our choice]**
- The prompt asks for exactly one word. The engine does not check it. **[our choice]**

**Debate**
- After the clues, everyone may talk. **[rulebook]** ("a few minutes")
- It is a talk phase for everyone, `talk_rounds` talk turns each (game config, default 1), in clue order. Talking is not required, so SILENCE is allowed. With `talk_rounds` 0 the phase does not happen. **[our choice]**

**Thinking**
- Everyone thinks at the very start, and again after the debate, before the vote. **[our choice]**

**Vote**
- Everyone votes for the player they think is the Chameleon, all at the same time, by pointing. **[rulebook]**
- A player votes for another player, never for itself. **[our choice]**
- The votes stay hidden until all are in, then they are made public together. **[rulebook]** (simultaneous pointing; the engine runs it as a simultaneous phase)
- The player with the most votes is accused. **[rulebook]**
- Tie: the dealer decides. **[rulebook]** It is one more decision for the dealer, choosing among the tied players. **[our choice]** The dealer may choose itself if it is one of the tied players. **[our choice, to confirm]**

**After the accusation**
- The accused turns over its card: the table learns whether it is the Chameleon. **[rulebook]**
- If the accused is not the Chameleon, the Chameleon escapes and the game ends. **[rulebook]**
- If the accused is the Chameleon, it guesses the secret word. A right guess means it escapes; a wrong guess means it is caught. **[rulebook]**
- The guess is one of the topic card's 16 words. **[our choice]**
- With 3 players the caught Chameleon gets two guesses; it stops at the first right one. **[rulebook]** The second guess cannot repeat the first, wrong, word. **[our choice, to confirm]**
- The secret word and the Chameleon are made public when the game ends. **[our choice]**

**Scoring.** The rulebook's optional score is 2 points for the Chameleon if it escapes undetected, 1 if it is caught but guesses right, and 2 for each other player if it is caught and guesses wrong. **[rulebook]** Rewards here are those scores halved, for one round: **[our choice]**

| Outcome | Chameleon | Each other player |
|---|---|---|
| Escapes undetected (the accused is not the Chameleon) | 1 | 0 |
| Caught, guesses right | 0.5 | 0 |
| Caught, guesses wrong | 0 | 1 |

**Step rewards** (the reward of one accepted step, for the player who took it): **[our choice]**
- A non-Chameleon's vote: 1 if it names the Chameleon, else 0.
- The Chameleon's guess: 1 if right, else 0.
- A non-Chameleon dealer's tie-break: 1 if it picks the Chameleon, else 0.
- Thinking steps: 0, as in `copycat_rps`.
- Every other step (the Chameleon's vote, a Chameleon dealer's tie-break, the clue and debate turns): the player's final reward.

## Phases, in the order they happen

Players are `p1` … `pN` in seat order. "Dealer order" is the dealer first, then clockwise. A phase that cannot happen is skipped.

| # | Phase | Mode | Eligible | talk_rounds | talk_required | Legal actions | Happens |
|---|---|---|---|---|---|---|---|
| 1 | `thinking` | THINKING | everyone, seat order | 0 | no | (thinking) | always |
| 2 | `clues` | DISCUSSION | everyone, dealer order | 1 | **yes** | `"done"` (one option, so the runner plays it without asking) | always |
| 3 | `debate` | DISCUSSION | everyone, dealer order | `talk_rounds` (default 1) | no | `"done"` (forced) | if `talk_rounds` > 0 |
| 4 | `thinking` | THINKING | everyone, seat order | 0 | no | (thinking) | always |
| 5 | `vote` | SIMULTANEOUS | everyone, seat order | 0 | no | `{"vote":"pK"}` for every other player | always |
| 6 | `tiebreak` | ACTION | the dealer | 0 | no | `{"choose":"pK"}` for every tied player | only if the top vote count is tied |
| 7 | `guess` | ACTION | the Chameleon | 0 | no | `{"guess":<word>}` for each of the 16 words (not a word already guessed) | only if the accused is the Chameleon; a second time with 3 players if the first guess was wrong |

The clue and debate words are table talk, not game actions: the game only sees the forced `"done"`; the runner shows every talk message to every later prompt.

**Observation.** Public: the players, the dealer, the topic and its 16 words, the phase, the turn order, and after the vote everyone's votes and the accused. Private: the secret word, for everyone except the Chameleon; whether you are the Chameleon, for you only. Hidden: the Chameleon's identity and the secret word from everyone who is not entitled to them, until the game ends.

**game_config.**

| Key | Default | Meaning |
|---|---|---|
| `seed` | (each task) | the dealer, the Chameleon, the topic and the secret word |
| `players` | (required) | 3 to 6 |
| `talk_rounds` | 1 | debate talk turns each; 0 skips the debate |

## Not implemented

- 7 and 8 players (the topic card is turned face down after the clues).
- Scoring across rounds, the first-to-5 target, the caught Chameleon dealing next, the dice and the Code Cards.
- A check that a clue is exactly one word, or that it differs from the topic's words.
