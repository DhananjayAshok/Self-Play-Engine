# CLAUDE GENERATED 100%, DJ DOES NOT APPROVE, MUST BE REVISED
"""Learning dynamics of copycat RPS RL runs: per training step, the orchestrator's reward and
truncation, the trainer's entropy, mismatch KL and grad norm, and (from the training games, one
batch per step) the invalid-answer rate and the exploit-move rate; then a one-line summary per
run with convergence speed.

    python scripts/run_dynamics.py --runs llama-s4-seg-unit-lr3e-5 llama-s5-lr1e-5 [--table]

Convergence: the first step from which the exploit rate stays >= 0.9 for 3 batches in a row.
"""

import json
import re
from pathlib import Path

import click

from project_utils import load_parameters

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Step (\d+) \|(.*)")
FIELDS = {
    "reward": re.compile(r"Reward (-?[0-9.]+)"),
    "trunc": re.compile(r"Truncation ([0-9.]+)%"),
    "entropy": re.compile(r"Entropy ([0-9.]+)"),
    "kl": re.compile(r"Mismatch KL ([0-9.]+)"),
    "grad": re.compile(r"Grad\. Norm ([0-9.]+)"),
}


def log_steps(*, path):
    """{step: {field: value}} from an orchestrator or trainer log."""
    steps = {}
    if not path.exists():
        return steps
    for line in path.read_text(errors="replace").splitlines():
        match = STEP.search(ANSI.sub("", line))
        if not match:
            continue
        values = steps.setdefault(int(match.group(1)), {})
        for name, pattern in FIELDS.items():
            found = pattern.search(match.group(2))
            if found:
                values[name] = float(found.group(1))
    return steps


STEP_TIME = re.compile(r"^(\d\d):(\d\d):(\d\d) .*Step (\d+) \|")
ROUND = re.compile(r"Round \d+: the player played (\w+), the opponent played (\w+); (.+)\.")
BEATEN_BY = {"scissors": "rock", "rock": "paper", "paper": "scissors"}


def step_end_times(*, path, first_arrival):
    """{step: unix time the orchestrator logged it}. The log has only times of day: they are
    placed on the date of the run's first game and rolled over midnight when they go backwards."""
    import datetime

    day = datetime.datetime.fromtimestamp(first_arrival).replace(hour=0, minute=0, second=0, microsecond=0)
    out, previous = {}, None
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_TIME.search(ANSI.sub("", line))
        if not match:
            continue
        h, m, s, step = (int(g) for g in match.groups())
        when = day + datetime.timedelta(hours=h, minutes=m, seconds=s)
        if previous is not None and when < previous:
            day += datetime.timedelta(days=1)
            when += datetime.timedelta(days=1)
        previous = when
        out[step] = when.timestamp()
    return out


def game_summary(*, row):
    traces = row.get("traces") or [row]
    info = traces[0].get("info", {}).get("game")
    if info is None:
        return None
    moves = [m.group(1) for e in info["record"] if e.get("kind") == "event"
             for m in [ROUND.fullmatch(e.get("content", ""))] if m]
    return {"invalid": info["invalid"]["player"] > 0,
            "exploit": [moves[i] == BEATEN_BY[moves[i - 1]] for i in range(1, len(moves))]}


def strategy_by_step(*, run_dir, orchestrator_log):
    """{step: (invalid rate, exploit rate)} over the training games that arrived before that
    step's batch was logged (and after the previous one's), plus "post" for games that arrived
    after the last step (generated with the final weights)."""
    import bisect
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from train_strategy import stream_lines

    traces = run_dir / "monitors/file/traces"
    if not (traces / "stream.index.jsonl").exists():
        return {}
    index = [json.loads(line) for line in (traces / "stream.index.jsonl").read_text().splitlines()]
    index = [row for row in index if row.get("kind") == "train"]
    if not index or not orchestrator_log.exists():
        return {}
    summaries = [game_summary(row=json.loads(line)) for line in stream_lines(run_dir=run_dir)]
    ends = step_end_times(path=orchestrator_log, first_arrival=min(row["arrival"] for row in index))
    steps = sorted(ends)
    times = [ends[s] for s in steps]
    grouped: dict = {}
    for row in index:
        game = summaries[row["line"] - 1] if row["line"] - 1 < len(summaries) else None
        if game is None:
            continue
        i = bisect.bisect_left(times, row["arrival"])
        grouped.setdefault(steps[i] if i < len(steps) else "post", []).append(game)
    out = {}
    for key, chunk in grouped.items():
        exploit = [e for g in chunk if not g["invalid"] for e in g["exploit"]]
        out[key] = (sum(g["invalid"] for g in chunk) / len(chunk), sum(exploit) / len(exploit) if exploit else float("nan"))
    return out


def convergence_step(*, exploit, threshold=0.9, run=3):
    for i in range(len(exploit) - run + 1):
        if all(e >= threshold for e in exploit[i:i + run]):
            return i + 1
    return None


@click.command()
@click.option("--runs", multiple=True, required=True, help="Run names under $storage_dir/rl_runs (repeat or space-separate)")
@click.option("--table", is_flag=True, help="Print the per-step table of every run")
def main(runs, table):
    parameters = load_parameters()
    root = Path(parameters["storage_dir"]) / "rl_runs"
    names = [n for r in runs for n in r.split()]
    summaries = []
    for name in names:
        run_dir = root / name
        orch_log = run_dir / "logs/latest/orchestrator.log"
        orch = log_steps(path=orch_log)
        train = log_steps(path=run_dir / "logs/latest/trainer.log")
        strategy = strategy_by_step(run_dir=run_dir, orchestrator_log=orch_log)
        nan = float("nan")
        n = max([*orch.keys(), *train.keys()], default=0)
        if table:
            print(f"== {name}")
            print("  step | reward |  trunc | entropy |     kl |  grad | invalid | exploit")
            for s in [*range(1, n + 1), "post"]:
                o, t = orch.get(s, {}), train.get(s, {})
                inv, ex = strategy.get(s, (nan, nan))
                print(f"  {s!s:>4} | {o.get('reward', nan):6.3f} | {o.get('trunc', nan):5.1f}% | "
                      f"{t.get('entropy', nan):7.3f} | {t.get('kl', nan):6.4f} | "
                      f"{t.get('grad', nan):5.3f} | {inv:7.2f} | {ex:7.2f}")
        exploit = [strategy.get(s, (nan, nan))[1] for s in range(1, n + 1)]
        invalid = [strategy.get(s, (nan, nan))[0] for s in range(1, n + 1)]
        conv = convergence_step(exploit=exploit)
        fixed = next((i + 1 for i, inv in enumerate(invalid) if inv <= 0.05), None)
        entropies = [train[s]["entropy"] for s in sorted(train) if "entropy" in train[s]]
        kls = [train[s]["kl"] for s in sorted(train) if "kl" in train[s]]
        post = strategy.get("post", (nan, nan))
        summaries.append((name, len(train), fixed, conv, max([e for e in exploit if e == e], default=nan),
                          exploit[-1] if exploit else nan, post[1], invalid[-1] if invalid else nan,
                          entropies[0] if entropies else nan, entropies[-1] if entropies else nan,
                          max(kls, default=nan)))
    print("run | trained steps | format fixed (invalid<=5%) at step | converged (exploit>=0.9 x3) at step | max exploit | "
          "last-step exploit | post-training exploit | last invalid | entropy first->last | max mismatch KL")
    for s in summaries:
        print(f"{s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]:.2f} | {s[5]:.2f} | {s[6]:.2f} | {s[7]:.2f} | "
              f"{s[8]:.2f}->{s[9]:.2f} | {s[10]:.4f}")


if __name__ == "__main__":
    main()
