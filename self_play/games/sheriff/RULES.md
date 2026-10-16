# Sheriff of Nottingham

Base game of the first edition (Arcane Wonders; designers Sérgio Halaban, André Zatz, Bryan Pope). 3 to 5 players. No optional rules: no Royal Goods, no seven-card hand.

## Sources

- First-edition rulebook, transcribed by RulesPal: https://rulespal.com/sheriff-of-nottingham/rulebook (every rule below)
- Second-edition rulebook, transcribed by RulesPal: https://rulespal.com/sheriff-of-nottingham-2nd-edition/rulebook (only for the differences listed at the end)

## Rules

**Cards.** The 3-player game leaves out every card marked 4+.

| Good | Kind | 3 players | 4–5 players | Value | Penalty |
|---|---|---|---|---|---|
| Apples | legal | 48 | 48 | 2 | 2 |
| Cheese | legal | 36 | 36 | 3 | 2 |
| Bread | legal | 0 | 36 | 3 | 2 |
| Chickens | legal | 24 | 24 | 4 | 2 |
| Pepper | contraband | 18 | 22 | 6 | 4 |
| Mead | contraband | 16 | 21 | 7 | 4 |
| Silk | contraband | 9 | 12 | 8 | 4 |
| Crossbow | contraband | 5 | 5 | 9 | 4 |

156 cards with 3 players, 204 with 4 or 5.

**Setup**
- Shuffle the goods into a draw pile and deal 6 cards to each player. Turn 5 cards face up onto a first discard pile, then 5 onto a second.
- Each player has 50 gold.

**A round** has one Sheriff; everyone else is a merchant.
1. **Market.** Each merchant in turn, clockwise from the Sheriff's left, may set aside up to 5 cards from their hand face down, then draws back to 6 cards, one at a time, from the top of either discard pile or the draw pile, in any combination. Every card wanted from the discard piles must be taken before any card from the draw pile. Then the set-aside cards go face up onto one discard pile of the merchant's choice, in any order. The Sheriff takes no part.
2. **Loading.** All merchants at the same time secretly put 1 to 5 cards from their hand into their bag. A closed bag cannot be changed.
3. **Declaration.** Each merchant in turn, clockwise from the Sheriff's left, declares the exact number of goods in the bag and exactly one type of legal good. The declaration may be false; contraband can never be declared.
4. **Inspection.** The Sheriff chooses which bags to inspect and in what order, one bag at a time, and may inspect none.
   - A bag that passes: legal goods go face up to the merchant's stand, contraband face down (only its number is shown).
   - An inspected honest bag: the Sheriff pays the merchant the penalty of every good in it; the goods go to the stand.
   - An inspected lying bag: the truthfully declared goods go to the stand; every other good is confiscated and placed on a discard pile, in the order the Sheriff chooses; the merchant pays the Sheriff the penalty of every confiscated good.
   - Bribes: a merchant may offer gold, legal goods or contraband from their stand, goods in the bag, and promises of future favours, but never cards in hand. Accepted deals must be honoured, except promises of future favours. Promised goods the merchant does not have need not be paid, but the merchant must show their cards to prove it. A Sheriff who accepts a bribe lets the bag pass; once the bag is returned or opened, negotiation over it ends. Bribe gold goes to the Sheriff's supply, bribe goods to the Sheriff's stand.
   - Paying without enough gold: give legal goods from the stand worth at least the debt (no change), then contraband; once the stand is empty, the rest of the debt is considered paid. The goods go to the receiver's stand.
5. **End of round.** All players draw from the draw pile back to 6 cards, and the Sheriff passes to the left.

**Draw pile.** When it runs out, shuffle all but the top five cards of each discard pile into a new draw pile. An emptied discard pile stays empty until a card is placed on it.

**Game length.** Each player is Sheriff three times with 3 players, twice with 4 or 5 (9, 8 or 10 rounds).

**Scoring.** Cards in hand score nothing. Each player scores the value of every good in their stand (legal and contraband), plus their gold, plus King and Queen bonuses:

| Legal good | King | Queen |
|---|---|---|
| Apples | 20 | 10 |
| Cheese | 15 | 10 |
| Bread | 15 | 10 |
| Chickens | 10 | 5 |

- The King has the most of that good in their stand, the Queen the second most.
- Tied for King: add the King and Queen bonuses and split them evenly among the tied players, rounded down; no Queen bonus is paid.
- Tied for Queen: split the Queen bonus evenly, rounded down.
- The most points wins. Ties go to the player with the most legal goods, then the most contraband.

## Implemented

This step: setup, the whole Market, loading, declaring, the Sheriff choosing the order of bags, a bag that passes, the end of the round (drawing back to 6, the rebuilt draw pile, the Sheriff passing left), the official number of rounds, and scoring with every bonus and tie rule.

Not yet (the next step): negotiation, bribes, inspection, confiscation and paying without enough gold. Until then **every bag passes**.

## How a round becomes phases

Each decision is one dialogue phase, and every big choice is a chain of small, official steps, so every legal list stays short. Players are `p1` … `pN` in seat order; merchants act clockwise from the Sheriff's left.

| Phase | Mode | Eligible | talk_rounds | Legal actions (the first is the runner's fallback: the honest or do-nothing one) | Closes when |
|---|---|---|---|---|---|
| `set_aside` | ACTION | the market merchant | 0 | `"done"`, then `{"set_aside":<good>}` per distinct good in hand; only `"done"` once 5 are set aside | each step; `"done"` moves on to the draws (or to the next merchant if nothing was set aside) |
| `draw` | ACTION | the same merchant | 0 | `{"draw":"deck"}`, `{"draw":"discard_1"}`, `{"draw":"discard_2"}` (a pile only if it has cards); after one draw from the draw pile, only `{"draw":"deck"}` | each step; the draws end at 6 cards, or when nothing is left to draw |
| `discard_pile` | ACTION | the same merchant | 0 | `{"pile":1}`, `{"pile":2}` | the pile is chosen |
| `discard_order` | ACTION | the same merchant | 0 | `{"place":<good>}` per distinct good still set aside; the last placed is the top. When one good is left, the rest is placed without asking | everything is placed: the next merchant's Market, or loading |
| `load` | SIMULTANEOUS | merchants still loading | 0 | empty bag: `{"load":<good>}` per distinct good in hand, legal goods first, the most numerous first; otherwise `"done"` first | every merchant still loading has answered for this card (same `phase_id` until then); a bag closes at `"done"`, at 5 cards, or with an empty hand |
| `declare` | ACTION | each merchant in turn | 0 | `{"declare":<legal good>}` per legal good in the deck (no Bread with 3 players), the bag's most numerous legal good first | the declaration is made; the game adds the count |
| `pick_bag` | ACTION | the Sheriff | 0 | `{"bag":<merchant>}` per bag not yet dealt with, clockwise from the Sheriff's left | the bag is dealt with (this step: it passes) |

**Automatic steps (no decision):** placing a single remaining good; passing a bag; the draws back to 6 at the end of a round (merchants clockwise from the Sheriff's left, then the Sheriff); rebuilding the draw pile (`rng(label="reshuffle", index=n)`); the Sheriff passing left; scoring.

**Observation.**
- **Public:** every player's gold, hand size and stand (legal goods by type, contraband by count); the market merchant's number of set-aside cards; declarations; the bags still to be dealt with; the draw pile's size; each discard pile's top card and size; the round, the Sheriff, who is deciding; and the game's public log.
- **Private:** the player's own hand, set-aside cards, bag, and the types of its face-down contraband.
- **Hidden:** all of those for other players, and the draw pile's order.
- **Loading:** while bags are loaded, every merchant who started loading is shown as deciding until all are done, so nobody learns when another bag closed.
- **The log** names every card taken from or put on a discard pile (they are face up) and the legal goods of a passed bag; draws from the draw pile are logged by count only.

**Thinking.** Everyone thinks at the opening (round 1) and again at the start of every later round (`"round"`, default). `"deal"` will also ask the merchant and the Sheriff before each negotiation (the next step); until then it behaves like `"round"`.

**Rewards.** The winner gets 1.0; tied winners share 1.0 equally. Step rewards are the engine default (each step earns its player's final reward). The outcome also holds every player's goods, gold, bonus and total.

**game_config.**

| Key | Default | Meaning |
|---|---|---|
| `seed` | (each task) | the deal, the setup discard piles, the first Sheriff and every reshuffle |
| `players` | (required) | 3 to 5 |
| `talk_rounds` | 2 | talk turns for the merchant and the Sheriff in each negotiation (the next step) |
| `thinking` | `"round"` | `"round"` or `"deal"`, see Thinking |

## Interpretation (not stated in the rulebook)

- **Gold is public.** The rulebook does not say whether gold is hidden.
- **The first Sheriff is chosen by the seed.** The rulebook picks the player with the most real cash on hand, and a random player on a tie.
- **King and Queen need at least one card.** A player with none of a good is never its King or Queen.
- **A tie after both tie-breakers is a shared win.** The first edition is silent; the second edition says so.
- **Discard piles show only their top card and size.** Earlier cards are known only from the log.
- **When nothing is left to draw** (the draw pile is empty and no discard pile has more than five cards), draws stop short of 6. A merchant with an empty hand loads no bag that round.
- **The Market's draws come before the set-aside cards are placed**, so a merchant can never take back its own set-aside cards in the same Market.

## Differences in the second edition (not used)

- 3 to 6 players; 6 players use the Deputy rules.
- One discard pile, made by the Sheriff from the set-aside cards; Market draws come only from the draw pile; the Sheriff chooses who starts the Market.
- When the draw pile runs out, the whole discard pile is shuffled into it.
- New optional rules, such as the Black Market.
- It states that a tie after the tie-breakers is a shared win.
