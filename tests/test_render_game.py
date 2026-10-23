import asyncio
import copy
import importlib.util
import json
from pathlib import Path

from click.testing import CliRunner

from self_play.chat_model import ScriptedChatModel
from self_play.games.chameleon import ChameleonGame
from self_play.runner import Runner

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_game.py"
_spec = importlib.util.spec_from_file_location("render_game", SCRIPT)
render_game = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_game)


def reply(*, answer, thinking="scratch"):
    return f"<thinking>{thinking}</thinking>\n<answer>{answer}</answer>\n[STOP]"


def scripted_row(*, seed=3, invalid_voter=False):
    """One traces.jsonl row, built the way self_play_verifiers/env.py builds it, from a Chameleon game played
    with scripted models: everyone votes for one innocent player, who votes for the Chameleon."""
    config = {"seed": seed, "players": 4}
    ids = ChameleonGame(game_config=config).player_ids
    chameleon = ChameleonGame(game_config=config).state["chameleon"]
    target = next(pid for pid in ("p1", "p2") if pid != chameleon)  # p1 or p2: see the invalid voter below
    votes = {pid: target for pid in ids if pid != target} | {target: chameleon}
    replies = {pid: [reply(answer="notes"), reply(answer=f"```{pid}```", thinking="my clue"), reply(answer="SILENCE"),
                     reply(answer="NO CHANGE"), reply(answer=json.dumps({"vote": votes[pid]}))] for pid in ids}
    if invalid_voter:
        # A voter's fallback is its first legal vote, the first other player: here the target, so the outcome holds.
        voter = "p1" if target == "p2" else "p2"
        replies[voter][4] = reply(answer="maybe")
    game = ChameleonGame(game_config=config)
    runner = Runner(game=game, models={pid: ScriptedChatModel(replies=replies[pid]) for pid in ids})
    result = asyncio.run(runner.run())
    record = {"game": game.name, "game_config": game.game_config, "seats": {pid: f"seat{i}" for i, pid in enumerate(ids)},
              "resumed": False, "outcome": result["outcome"], "steps": game.steps, "step_rewards": result["step_rewards"],
              "invalid": result["invalid"], "record": runner.record}
    traces = [{"agent": {"config": {"model": f"model-{pid}"}}, "timing": {"agent": {"start": 100.0 + i, "end": 160.0 + i}},
               "info": {"game": {**record, "player_id": pid}}} for i, pid in enumerate(ids)]
    row = {"ok": True, "task": {"data": {"info": {"seed": seed}}}, "traces": traces}
    return json.loads(json.dumps(row)), runner, game


def test_a_game_is_rendered_with_a_header_and_every_record_entry_in_order():
    row, runner, game = scripted_row()
    markdown, problems = render_game.render_game(row=row)
    assert problems == []
    calls = [e for e in runner.record if e["kind"] == "call"]
    assert "# chameleon: seed 3" in markdown
    assert "p1 (model-p1), p2 (model-p2), p3 (model-p3), p4 (model-p4)" in markdown
    assert f"**Model calls:** {len(calls)} (8 think, 8 talk, 4 answer)" in markdown
    assert '**Invalid answers:** `{"p1": 0, "p2": 0, "p3": 0, "p4": 0}`' in markdown
    assert "**Wall time:** 1 min 3 s" in markdown  # the latest end (163) minus the earliest start (100)
    assert "**Replay check:** the replay agrees with the record" in markdown
    assert json.dumps(runner.game.get_outcome()["rewards"]) in markdown
    order = [markdown.index(f"## Phase {name}") for name in ("thinking", "clues", "debate", "vote")]
    assert order == sorted(order)
    assert markdown.count("### Call ") == len(calls)
    assert markdown.count("Game state after step") == game.n_steps == len(game.steps)
    assert markdown.count("<details>") == len(calls) + game.n_steps  # a prompt and a state each
    first_prompt = calls[0]["messages"][-1]["content"]
    assert first_prompt in markdown
    assert "**Thinking:**" in markdown and "my clue" in markdown
    assert "**Forced**" in markdown and "**Thought**" in markdown and "**Accepted**" in markdown
    assert "> **Event:** Votes: " in markdown and "says:** ```p" in markdown
    assert "````" in markdown  # a clue is one word inside a code fence, so the fences around it are longer


def test_the_state_after_each_step_is_rebuilt_from_the_seed():
    row, _, game = scripted_row()
    markdown, _ = render_game.render_game(row=row)
    blocks = markdown.split("Game state after step ")[1:]
    assert len(blocks) == game.n_steps
    assert '"stage": "clues"' in blocks[0] and '"stage": "over"' in blocks[-1]
    assert '"result": "escaped_undetected"' in blocks[-1]
    assert '"pending_thinking": [\n    "p2"' in blocks[0]  # after p1's first thinking turn


def test_rejected_and_fallback_answers_are_shown_with_their_error():
    row, _, _ = scripted_row(invalid_voter=True)
    markdown, problems = render_game.render_game(row=row)
    assert problems == []
    assert "**Rejected**" in markdown and "**Fallback**" in markdown
    error = next(line for line in markdown.splitlines() if line.startswith("- **Rejected**")).split("): ", 1)[1]
    assert "is not legal now" in error
    assert f"played because the answers failed: {error}" in markdown
    invalid = row["traces"][0]["info"]["game"]["invalid"]
    assert sum(invalid.values()) == 1 and f"**Invalid answers:** `{json.dumps(invalid)}`" in markdown


def test_a_record_that_disagrees_with_the_replay_is_reported():
    row, _, _ = scripted_row()
    wrong_event = copy.deepcopy(row)
    event = next(e for e in wrong_event["traces"][0]["info"]["game"]["record"] if e["kind"] == "event" and e["content"].startswith("Votes:"))
    event["content"] = "Votes: nobody voted."
    markdown, problems = render_game.render_game(row=wrong_event)
    assert any("public events" in problem for problem in problems)
    assert "DISAGREEMENT(S)" in markdown and "## Replay check\n- " in markdown
    wrong_outcome = copy.deepcopy(row)
    wrong_outcome["traces"][0]["info"]["game"]["outcome"]["result"] = "caught"
    markdown, problems = render_game.render_game(row=wrong_outcome)
    assert any("outcome" in problem for problem in problems) and "DISAGREEMENT(S)" in markdown


def test_a_run_gets_one_file_per_game_and_failed_games_are_skipped(tmp_path):
    run_dir, out_dir = tmp_path / "run-001", tmp_path / "out"
    run_dir.mkdir()
    rows = [scripted_row(seed=3)[0], {"ok": False, "traces": []}, scripted_row(seed=5)[0], scripted_row(seed=3)[0]]
    (run_dir / "traces.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = CliRunner().invoke(render_game.main, ["--run", str(run_dir), "--out_dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in out_dir.iterdir()) == ["seed3-2.md", "seed3.md", "seed5.md"]
    assert "line 2: no game record" in result.output and "3 game(s) written" in result.output
    assert (out_dir / "seed5.md").read_text().startswith("# chameleon: seed 5")
