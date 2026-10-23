"""One Markdown file per game of an eval run, to read a game from start to finish.

    python scripts/render_game.py --run <run_dir or traces.jsonl> [--out_dir <dir>]

Reads the run's traces.jsonl and writes <out_dir>/seed<seed>.md for every game that has a record
(default out_dir: <results_dir>/rendered/<run name>). Each file holds a header (game, seed, players
and their models, outcome, rewards, calls, invalid count, wall time), then the record in order:
phases, model calls (the prompt inside a collapsible block, the thinking, the answer), actions
(accepted, forced, fallback with its error, or rejected), public events and table talk. After
every accepted step it shows the full game state, rebuilt by replaying the accepted actions
through the game from its seed. Any disagreement between the replay and the record is written in
the file (and counted in the header).
"""

import json
from pathlib import Path

import click

from self_play.game import THINK
from self_play.games import GAMES


def fence(*, text, language=""):
    """A code fence longer than any run of backticks inside the text."""
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{language}\n{text}\n{ticks}"


def details(*, summary, body):
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def duration(*, seconds):
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def game_record_of(*, row):
    """The game record carried by the row's seat traces, or None for a failed game."""
    for trace in row.get("traces") or []:
        game = (trace.get("info") or {}).get("game")
        if game is not None:
            return game
    return None


def players_and_models(*, row):
    """{player_id: model} from the seat traces."""
    models = {}
    for trace in row.get("traces") or []:
        game = (trace.get("info") or {}).get("game")
        if game is not None:
            models[game["player_id"]] = ((trace.get("agent") or {}).get("config") or {}).get("model", "?")
    return models


def wall_time(*, row):
    """The latest timing.agent.end minus the earliest timing.agent.start over the seat traces."""
    agents = [(trace.get("timing") or {}).get("agent") for trace in row.get("traces") or []]
    agents = [a for a in agents if a and "start" in a and "end" in a]
    return max(a["end"] for a in agents) - min(a["start"] for a in agents) if agents else None


class Replay:
    """The game rebuilt from its seed, stepped with the record's accepted actions."""

    def __init__(self, *, record):
        self.problems = []
        self.game = None
        name = record["game"]
        if name not in GAMES:
            self.problems.append(f"unknown game {name!r}: no state can be rebuilt")
            return
        try:
            self.game = GAMES[name](game_config=record["game_config"])
        except Exception as error:  # a config the current code rejects
            self.problems.append(f"the game could not be rebuilt from its config: {error}")

    def step(self, *, entry, events):
        """Apply one accepted action; the state text, or None if the replay cannot go on."""
        if self.game is None:
            return None
        result = self.game.step(player_id=entry["player_id"], action=entry["action"])
        where = f"step {entry['step_id']} ({entry['player_id']} {json.dumps(entry['action'])})"
        if not result.accepted:
            self.problems.append(f"{where}: recorded as accepted, but the replay rejects it: {result.error}")
            self.game = None
            return None
        if self.game.n_steps - 1 != entry["step_id"]:
            self.problems.append(f"{where}: the replay made it step {self.game.n_steps - 1}")
        if result.public_events != events:
            self.problems.append(f"{where}: the replay's public events {result.public_events} differ from the recorded {events}")
        return json.dumps({"state": self.game.state, "pending_thinking": self.game.pending_thinking}, indent=2)

    def finish(self, *, record):
        if self.game is None:
            return
        if json.loads(json.dumps(self.game.steps)) != record["steps"]:
            self.problems.append("the replayed steps differ from the recorded steps")
        outcome = json.loads(json.dumps(self.game.get_outcome()))
        if outcome != record["outcome"]:
            self.problems.append(f"the replayed outcome {outcome} differs from the recorded {record['outcome']}")
        if outcome["done"] and json.loads(json.dumps(self.game.get_step_rewards())) != record["step_rewards"]:
            self.problems.append("the replayed step rewards differ from the recorded ones")


def render_call(*, number, entry):
    flag = "" if entry["answer"] is not None else " (no <answer> tag: unreadable)"
    parts = [f"### Call {number}: {entry['player_id']} · {entry['call']}{flag}",
             details(summary=f"What {entry['player_id']} saw (user message)", body=fence(text=entry["messages"][-1]["content"])),
             "**Thinking:**\n\n" + (fence(text=entry["thinking"]) if entry["thinking"] is not None else "_none_"),
             "**Answer:**\n\n" + (fence(text=entry["answer"]) if entry["answer"] is not None else "_none_")]
    return "\n\n".join(parts)


def render_action(*, entry, step_rewards, last_error):
    who, action = entry["player_id"], json.dumps(entry["action"])
    if not entry["accepted"]:
        return f"- **Rejected** {who}: `{action}` ({entry['source']}): {entry['error']}"
    step = entry["step_id"]
    reward = step_rewards[step] if step is not None and step < len(step_rewards) else None
    suffix = f" (step {step}" + (f", step reward {reward}" if reward is not None else "") + ")"
    if entry["source"] == "fallback":
        return f"- **Fallback** {who}: `{action}`, played because the answers failed: {last_error.get(who, 'no error recorded')}{suffix}"
    if entry["source"] == "forced":
        return f"- **Forced** {who}: `{action}`, the only legal action{suffix}"
    if entry["action"] == THINK:
        return f"- **Thought** {who}{suffix}"
    return f"- **Accepted** {who}: `{action}`{suffix}"


def render_game(*, row):
    """(markdown, problems) for one traces.jsonl row with a game record."""
    record = game_record_of(row=row)
    seed = record["game_config"].get("seed", "?")
    models = players_and_models(row=row)
    entries = record["record"]
    calls = [e for e in entries if e["kind"] == "call"]
    kinds = {kind: sum(1 for e in calls if e["call"] == kind) for kind in ("think", "talk", "answer")}
    seconds = wall_time(row=row)
    replay = Replay(record=record)
    body, number, last_error, index = [], 0, {}, 0
    while index < len(entries):
        entry = entries[index]
        kind = entry["kind"]
        index += 1
        if kind == "phase":
            talk = f", talk_rounds {entry['talk_rounds']}" + (", talk required" if entry.get("talk_required") else "")
            body.append(f"## Phase {entry['phase']} (id {entry['phase_id']}): {entry['mode']}, "
                        f"eligible {', '.join(entry['eligible'])}{talk}")
        elif kind == "call":
            number += 1
            body.append(render_call(number=number, entry=entry))
        elif kind == "message":
            body.append(f"> **{entry['player_id']} says:** {entry['content']}")
        elif kind == "event":
            body.append(f"> **Event:** {entry['content']}")
        elif kind == "action":
            if not entry["accepted"]:
                last_error[entry["player_id"]] = entry["error"]
            body.append(render_action(entry=entry, step_rewards=record["step_rewards"], last_error=last_error))
            if entry["accepted"]:
                events = []
                while index < len(entries) and entries[index]["kind"] == "event":
                    events.append(entries[index]["content"])
                    body.append(f"> **Event:** {entries[index]['content']}")
                    index += 1
                state = replay.step(entry=entry, events=events)
                if state is not None:
                    body.append(details(summary=f"Game state after step {entry['step_id']}", body=fence(text=state, language="json")))
                last_error.pop(entry["player_id"], None)
    replay.finish(record=record)
    outcome = record["outcome"]
    header = [
        f"# {record['game']}: seed {seed}",
        f"- **Game config:** `{json.dumps(record['game_config'])}`",
        "- **Players:** " + ", ".join(f"{pid} ({models.get(pid, '?')})" for pid in sorted(record["invalid"])),
        f"- **Outcome:** `{json.dumps({k: v for k, v in outcome.items() if k != 'rewards'})}`",
        f"- **Rewards:** `{json.dumps(outcome['rewards'])}`",
        f"- **Model calls:** {len(calls)} ({kinds['think']} think, {kinds['talk']} talk, {kinds['answer']} answer)",
        f"- **Invalid answers:** `{json.dumps(record['invalid'])}`",
        f"- **Wall time:** {duration(seconds=seconds) if seconds is not None else 'unknown'}",
        f"- **Replay check:** " + ("the replay agrees with the record" if not replay.problems
                                   else f"**{len(replay.problems)} DISAGREEMENT(S)**, see the end of this file"),
    ]
    tail = ["## Replay check"] + ([f"- {problem}" for problem in replay.problems] or ["The replay agrees with the record."])
    return "\n\n".join(["\n".join(header), *body, "\n".join(tail)]) + "\n", replay.problems


def render_run(*, traces_path, out_dir):
    """Write one file per game; the list of (file name, disagreement count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with open(traces_path) as f:
        for number, line in enumerate(f, start=1):
            row = json.loads(line)
            if game_record_of(row=row) is None:
                click.echo(f"line {number}: no game record (the game failed), skipped")
                continue
            markdown, problems = render_game(row=row)
            seed = game_record_of(row=row)["game_config"].get("seed", "x")
            name, copy = f"seed{seed}.md", 1
            while any(name == done for done, _ in written):
                copy += 1
                name = f"seed{seed}-{copy}.md"
            (out_dir / name).write_text(markdown)
            written.append((name, len(problems)))
    return written


@click.command()
@click.option("--run", "run", required=True, help="An eval run directory (with traces.jsonl) or a traces.jsonl file")
@click.option("--out_dir", default=None, help="Where the Markdown files go (default <results_dir>/rendered/<run name>)")
def main(run, out_dir):
    path = Path(run)
    traces_path = path if path.is_file() else path / "traces.jsonl"
    if out_dir is None:
        from project_utils import load_parameters
        parameters = load_parameters()
        out_dir = Path(parameters["results_dir"]) / "rendered" / traces_path.parent.name
    written = render_run(traces_path=traces_path, out_dir=Path(out_dir))
    for name, problems in written:
        click.echo(f"{Path(out_dir) / name}" + (f"  ({problems} replay disagreement(s))" if problems else ""))
    click.echo(f"{len(written)} game(s) written to {out_dir}")


if __name__ == "__main__":
    main()
