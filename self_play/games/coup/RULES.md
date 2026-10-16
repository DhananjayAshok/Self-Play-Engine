# Coup

Base game only (no Reformation, no expansions). 2 to 6 players.

## Sources

Two independent transcriptions of the official rulebook (designer Rikki Tahta). They agree on every rule below.

- 64 Oz. Games accessibility transcription: https://www.64ouncegames.com/pages/coup
- UltraBoardGames transcription: https://www.ultraboardgames.com/coup/game-rules.php

## Rules

**Setup**
- 15 cards: 3 each of Duke, Assassin, Captain, Ambassador, Contessa.
- Each player gets 2 face-down cards and 2 coins. The rest of the cards form the Court deck. Coins are public.
- With 2 players, the starting player gets only 1 coin.

**Turn**
- Clockwise. Exactly one action per turn. A player may not pass.
- A player who starts the turn with 10 or more coins must launch a Coup.

**Actions**

| Action | Effect | Character claimed | Can be blocked by |
|---|---|---|---|
| Income | +1 coin | none | nobody |
| Foreign Aid | +2 coins | none | any player claiming the Duke |
| Coup | pay 7, target loses 1 influence | none | nobody |
| Tax | +3 coins | Duke | nobody |
| Assassinate | pay 3, target loses 1 influence | Assassin | the target, claiming the Contessa |
| Steal | take 2 from the target (1 if they have 1) | Captain | the target, claiming the Captain or the Ambassador |
| Exchange | draw 2 from the Court deck, keep as many cards as you had, return the rest | Ambassador | nobody |

**Challenges**
- Any action or block that claims a character can be challenged, by any other player, even one who is not involved.
- Challenges are resolved before blocks. A challenge cannot be made after play has moved on.
- The challenged player proves the claim by showing the card. If they cannot, they lose the challenge.
- Whoever loses the challenge loses 1 influence immediately.
- If the claimant wins, they shuffle the shown card back into the Court deck and draw a replacement. Then the action or block goes ahead.

**Coins after a failed action**
- Action successfully **challenged**: the action fails and **its cost is refunded**.
- Action successfully **blocked**: the action fails and **its cost stays spent**.
- So an Assassin caught bluffing gets the 3 coins back, but an assassination blocked by the Contessa does not.

**Losing influence and elimination**
- The player losing influence chooses which card to reveal. Revealed cards stay face up.
- A player with no face-down cards is out and returns all coins to the Treasury.
- A player can lose 2 influence in one assassination turn: a lost challenge plus the assassination, or a bluffed Contessa that is challenged plus the assassination.

**Talk**
- Negotiation is allowed but never binding. Players may not show cards to each other. Coins cannot be given or lent.

## Implemented so far

2 to 6 players; Income, Foreign Aid (not yet blockable) and Coup; the seeded deal, hidden hands, choosing which influence to lose, elimination and the winner. Character actions, challenges, blocks and the Exchange come in later steps.

Turns go clockwise (seat order `p1` … `pN`) from the starting player, who is chosen by the seed. Eliminated players are skipped in turns and are never asked again. Any living player other than the actor may be targeted.

## How a turn becomes phases

Each decision is one dialogue phase. Players are `p1` … `pN` in seat order.

| Phase | Mode | Eligible | talk_rounds | Legal actions (the first is the runner's fallback) | Closes when | Status |
|---|---|---|---|---|---|---|
| `action` | ACTION | the actor | `talk_rounds` | at 10+ coins only `{"action":"coup","target":…}`; otherwise `{"action":"income"}`, `{"action":"foreign_aid"}`, then `{"action":"coup","target":…}` per living opponent at 7+ coins (`tax`, `exchange`, `steal`, `assassinate` later) | the action is applied | done |
| `challenge_action` | RESPONSE_WINDOW | living non-actors, clockwise from the actor | `window_talk_rounds` | `"pass"`, `"challenge"` | the first challenge, or everyone passed | later |
| `block_action` | RESPONSE_WINDOW | Foreign Aid: every living non-actor; Steal, Assassinate: the target | `window_talk_rounds` | `"pass"`, then `{"block":<card>}` per allowed card | the first block, or everyone passed | later |
| `challenge_block` | RESPONSE_WINDOW | living players except the blocker, clockwise from the blocker | `window_talk_rounds` | `"pass"`, `"challenge"` | the first challenge, or everyone passed | later |
| `lose_influence` | ACTION | the first player in the loss queue | 0 | `{"reveal":<card>}` per distinct face-down card | the card is revealed (the runner plays a single option itself) | done |
| `exchange` | ACTION | the Ambassador | 0 | `{"keep":[…]}` per distinct sorted set, current hand first | the set is kept | later |

A pass keeps the window's `phase_id` and removes the passer from `eligible`; a challenge or block opens a new phase, so the runner asks nobody else.

**Observation.** Public: every player's coins, revealed cards, number of face-down cards and whether they are in; the deck size; the turn, the actor and who is deciding; the game's public log. Private: the player's own face-down cards. Hidden: other players' face-down cards and the deck. The game keeps its own log because the runner never puts its `event` entries in prompts.

**Thinking.** Everyone thinks once at the opening. After that, `thinking` decides who thinks when a turn ends:
- `"actor"` (default, option B): only the next actor, at the start of its turn. One extra call per turn; responders still see the full public log.
- `"all"` (option A): every living player. One extra call per living player per turn; notes are always current.

**Rewards.** The last player with influence gets 1.0, everyone else 0.0. Step rewards are the engine default (each step earns its player's final reward).

**game_config.**

| Key | Default | Meaning |
|---|---|---|
| `seed` | (each task) | the deal and the starting player |
| `players` | (required) | 2 to 6 |
| `max_turns` | 100 | the turn limit |
| `talk_rounds` | 1 | talk turns before the main action |
| `window_talk_rounds` | 0 | talk turns before each answer in challenge and block windows (the windows arrive with challenges and blocks) |
| `thinking` | `"actor"` | `"actor"` or `"all"`, see Thinking |

## Interpretation (not stated in the rulebook)

- **Serialised phases.** Because challenges come before blocks, a player who loses a challenge against an action may still block it afterwards, if they have influence left.

## Simplifications

- A challenged player who holds the claimed card always reveals it.
- Talk is attached to decisions (`talk_rounds` before the main action, `window_talk_rounds` in windows); there is no free chat.
- The Treasury is unlimited.
- The starting player is chosen by the seed. The rulebook says the winner of the previous game starts.
- The optional 2-player drafting variant is not implemented. The basic 2-player rule (the starting player gets 1 coin) is.
- When a player has exactly one legal action (for example revealing their last card), the runner applies it without asking.
- **Turn limit (not a rule).** A game that reaches `max_turns` turns ends without a winner (`termination_reason: "turn_limit"`), every reward 0, so looping agents cannot play forever.
