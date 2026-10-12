# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
"""Paired comparison of two copycat RPS eval runs (vf-eval traces.jsonl) on the same seeds.

    python scripts/compare_evals.py --base <run_dir> --treat <run_dir>
    python scripts/compare_evals.py --base <dir1> --base <dir2> --treat <dir1> --treat <dir2>   # pooled

Per game: the game reward (1 win, 0.5 draw, 0 loss on rounds won vs lost), each round's
winner, and whether any answer was invalid (for the forfeit-adjusted mean, which scores games
the way `env.forfeit_on_invalid` does in training). Reports means, a paired bootstrap 95% CI for the reward difference, per-seed
better/same/worse counts, and the per-round win rate (rounds 2+ are where copying the
player's last move can be exploited).
"""

import json
import random
import re
from pathlib import Path

import click


def load_games(*, run_dir):
    games = {}
    with open(Path(run_dir) / "traces.jsonl") as f:
        for line in f:
            row = json.loads(line)
            seed = row["task"]["data"]["info"]["seed"]
            info = row["traces"][0].get("info", {}).get("game") if row.get("ok") and row.get("traces") else None
            if info is None:
                games[seed] = None
                continue
            winners, moves = _rounds(record=info["record"])
            games[seed] = {
                "reward": info["outcome"]["rewards"]["player"],
                "invalid": info["invalid"]["player"],
                "winners": winners,
                "moves": moves,
            }
    return games


BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
BEATEN_BY = {loser: winner for winner, loser in BEATS.items()}
ROUND_EVENT = re.compile(r"Round \d+: the player played (\w+), the opponent played (\w+); (.+)\.")


def _rounds(*, record):
    """Each round's winner and the player's move, from the game's 'event' records."""
    winners, moves = [], []
    for entry in record:
        match = ROUND_EVENT.fullmatch(entry.get("content", "")) if entry.get("kind") == "event" else None
        if match is None:
            continue
        moves.append(match.group(1))
        winners.append({"the player wins": "player", "the opponent wins": "opponent"}.get(match.group(3)))
    return winners, moves


@click.command()
@click.option("--base", required=True, multiple=True, help="Baseline eval run dir (repeat to pool runs over disjoint seeds)")
@click.option("--treat", required=True, multiple=True, help="Trained eval run dir (repeat to pool runs over disjoint seeds)")
@click.option("--bootstrap", default=10000, help="Bootstrap resamples")
@click.option("--seed", default=0, help="Bootstrap RNG seed")
def main(base, treat, bootstrap, seed):
    base_games, treat_games = {}, {}
    for run_dir in base:
        base_games.update(load_games(run_dir=run_dir))
    for run_dir in treat:
        treat_games.update(load_games(run_dir=run_dir))
    seeds = sorted(s for s in base_games if s in treat_games and base_games[s] and treat_games[s])
    failed = {"base": sum(g is None for g in base_games.values()), "treat": sum(g is None for g in treat_games.values())}
    print(f"paired games: {len(seeds)} (failed/unscored: {failed})")

    for name, games in (("base", base_games), ("treat", treat_games)):
        rewards = [games[s]["reward"] for s in seeds]
        forfeited = [0.0 if games[s]["invalid"] else games[s]["reward"] for s in seeds]
        print(
            f"{name:5s}: mean reward {sum(rewards) / len(rewards):.3f} | games won {sum(r == 1.0 for r in rewards)}, "
            f"drawn {sum(r == 0.5 for r in rewards)}, lost {sum(r == 0.0 for r in rewards)}"
        )
        # As the training reward scores it with env.forfeit_on_invalid: any invalid answer -> 0.
        print(
            f"       forfeit-adjusted mean {sum(forfeited) / len(forfeited):.3f} "
            f"(games with an invalid answer: {sum(bool(games[s]['invalid']) for s in seeds)})"
        )
        max_rounds = max(len(games[s]["winners"]) for s in seeds)
        per_round = []
        for i in range(max_rounds):
            results = [games[s]["winners"][i] for s in seeds if len(games[s]["winners"]) > i]
            per_round.append(
                f"r{i + 1} W{sum(w == 'player' for w in results) / len(results):.2f}/"
                f"L{sum(w == 'opponent' for w in results) / len(results):.2f}"
            )
        print("       round win/loss rates:", "  ".join(per_round))
        # The exploit: from round 2 on, play what beats your own previous move.
        exploit = [
            games[s]["moves"][i] == BEATEN_BY.get(games[s]["moves"][i - 1])
            for s in seeds
            for i in range(1, len(games[s]["moves"]))
        ]
        print(f"       exploit move rate (rounds 2+): {sum(exploit) / max(len(exploit), 1):.2f}")

    diffs = [treat_games[s]["reward"] - base_games[s]["reward"] for s in seeds]
    mean_diff = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(bootstrap))
    low, high = boots[int(0.025 * bootstrap)], boots[int(0.975 * bootstrap) - 1]
    print(f"reward diff (treat - base): {mean_diff:+.3f}, paired bootstrap 95% CI [{low:+.3f}, {high:+.3f}]")
    print(
        f"per seed: treat better {sum(d > 0 for d in diffs)}, same {sum(d == 0 for d in diffs)}, "
        f"worse {sum(d < 0 for d in diffs)}"
    )


if __name__ == "__main__":
    main()
