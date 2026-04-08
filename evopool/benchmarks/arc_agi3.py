"""
ARC-AGI-3 Adapter for EvoPool.

Wraps ARC-AGI-3 interactive environments as EvoPool tasks.
Each environment becomes a multi-round "task" where agents take turns
exploring, reasoning, and acting in the environment.

Key design:
- Agent pool plays each environment multiple times (warmup + evaluated runs)
- Each "play" is a sequence of (observe grid → reason → pick action) steps
- CoDream happens BETWEEN plays: agents share discovered rules/strategies
- Score = levels_completed / total_levels (RHAE-inspired)
"""

import json
import re
import time
import numpy as np

import arc_agi
from arcengine.enums import GameAction, GameState


# ---- Grid rendering for LLM ----

_COLOR_CHARS = {
    0: '.', 1: 'R', 2: 'G', 3: 'B', 4: '_', 5: 'Y', 6: 'M', 7: 'C',
    8: 'O', 9: 'P', 10: 'W', 11: 'K', 12: 'L', 13: 'N', 14: 'T', 15: 'S'
}

def grid_to_text(frame: np.ndarray) -> str:
    """Convert 64x64 grid to full-resolution text. 1 char per cell, no downsampling."""
    lines = []
    for i, row in enumerate(frame):
        chars = ''.join(_COLOR_CHARS.get(int(c), '?') for c in row)
        lines.append(chars)
    return "\n".join(lines)


def describe_changes(prev: np.ndarray, curr: np.ndarray) -> str:
    """Describe what changed between two frames on full grid."""
    diff_r, diff_c = np.where(prev != curr)
    if len(diff_r) == 0:
        return "No visible change."
    n = len(diff_r)
    samples = [(int(diff_r[i]), int(diff_c[i])) for i in range(min(8, n))]
    parts = [f"({r},{c}): {_COLOR_CHARS.get(int(prev[r,c]),'?')}→{_COLOR_CHARS.get(int(curr[r,c]),'?')}" for r, c in samples]
    return f"{n} pixels changed: {', '.join(parts)}"


# ---- Single agent play session ----

def agent_play(
    env,
    llm_call_fn,
    model: str,
    max_steps: int = 200,
    experience_context: str = "",
    agent_id: str = "agent",
) -> dict:
    """
    One agent plays one environment for up to max_steps.

    Returns:
        {"levels_completed": int, "total_levels": int, "score": float,
         "actions_taken": int, "trajectory_summary": str,
         "discovered_rules": list[str]}
    """
    obs = env.reset()
    d = obs.model_dump()
    frame = obs.frame[0]
    available = d['available_actions']
    total_levels = d['win_levels']

    trajectory = []
    prev_frame = frame.copy()
    prev_levels = 0

    for step in range(max_steps):
        grid_text = grid_to_text(frame)
        changes = describe_changes(prev_frame, frame) if step > 0 else "First observation."

        # Build prompt
        history_text = ""
        if trajectory:
            recent = trajectory[-5:]  # last 5 actions
            history_text = "Recent actions:\n" + "\n".join(
                f"  Step {t['step']}: action={t['action']} → {t['result']}"
                for t in recent
            ) + "\n\n"

        # ACTION6 requires x,y coordinates. Describe available actions properly.
        action_descriptions = []
        for a in available:
            if a == 6:
                action_descriptions.append("6 (click at x,y — requires coordinates)")
            elif a == 7:
                action_descriptions.append("7 (undo)")
            else:
                action_descriptions.append(str(a))
        action_desc = ", ".join(action_descriptions)

        # For ACTION6 games, ask for coordinates too
        has_click = 6 in available
        if has_click:
            action_format = (
                '{"action": <int>, "x": <0-63>, "y": <0-63>, "reasoning": "brief why"}\n'
                'For action 6 (click), x and y are required grid coordinates.'
            )
        else:
            action_format = '{"action": <int>, "reasoning": "brief why"}'

        # Send full grid every 5 steps; otherwise just send changes to save context
        if step % 5 == 0:
            grid_section = f"Grid (64x64):\n{grid_text}\n"
        else:
            grid_section = f"Grid changes since last action: {changes}\n"

        # Build clear action instruction
        if has_click and len(available) == 1 and available[0] == 6:
            # Click-only game
            action_instruction = (
                "This is a CLICK game. You MUST click on a specific grid cell.\n"
                "Reply JSON: {\"action\": 6, \"x\": <column 0-63>, \"y\": <row 0-63>, \"reasoning\": \"why\"}\n"
                "Look at the grid and choose WHERE to click. Try clicking on colored objects or patterns."
            )
        elif has_click:
            action_instruction = (
                f"Available actions: {action_desc}\n"
                "For action 6 (click), specify coordinates: {\"action\": 6, \"x\": <col>, \"y\": <row>}\n"
                "For other actions: {\"action\": <number>}\n"
                "Reply with JSON only."
            )
        else:
            action_instruction = (
                f"Available actions: {action_desc}\n"
                "Reply JSON: {\"action\": <number>, \"reasoning\": \"why\"}"
            )

        prompt = (
            f"You are playing an abstract puzzle game. Explore and figure out the rules.\n\n"
            f"{grid_section}\n"
            f"Levels: {d['levels_completed']}/{total_levels} | Step: {step}/{max_steps}\n\n"
            f"{history_text}"
            f"{experience_context}"
            f"{action_instruction}"
        )

        try:
            response = llm_call_fn(
                model=model, user=prompt, max_tokens=150, enable_thinking=False
            )
            action_num, click_x, click_y = _parse_action_with_coords(response, available)
        except Exception:
            action_num = available[0]
            click_x, click_y = 32, 32  # center default

        # Execute action
        _ACTION_MAP = {a.value: a for a in GameAction}
        action = _ACTION_MAP.get(action_num, _ACTION_MAP.get(available[0], GameAction.ACTION1))

        try:
            if action_num == 6:
                obs = env.step(action, data={"x": click_x, "y": click_y})
            else:
                obs = env.step(action)
        except Exception:
            try:
                obs = env.reset()
            except Exception:
                break

        if obs is None:
            break

        d = obs.model_dump()
        prev_frame = frame.copy()
        frame = obs.frame[0] if obs.frame else frame
        new_levels = d['levels_completed']

        result_str = describe_changes(prev_frame, frame)
        if new_levels > prev_levels:
            result_str += f" LEVEL {new_levels} COMPLETED!"

        trajectory.append({
            "step": step,
            "action": action_num,
            "reasoning": response[:100] if response else "",
            "result": result_str,
            "levels": new_levels,
        })

        prev_levels = new_levels
        state = d['state']
        if str(state) in ('GameState.GAME_OVER', 'GameState.WIN'):
            break

    # Generate trajectory summary
    summary = _summarize_trajectory(trajectory, llm_call_fn, model)

    return {
        "levels_completed": d['levels_completed'],
        "total_levels": total_levels,
        "score": d['levels_completed'] / total_levels if total_levels > 0 else 0.0,
        "actions_taken": len(trajectory),
        "trajectory_summary": summary,
        "trajectory": trajectory,
    }


def _parse_action(response: str, available: list[int]) -> int:
    """Extract action number from LLM response."""
    try:
        m = re.search(r'\{[^}]*"action"\s*:\s*(\d+)', response)
        if m:
            act = int(m.group(1))
            if act in available:
                return act
    except Exception:
        pass
    for ch in response:
        if ch.isdigit() and int(ch) in available:
            return int(ch)
    return available[0]


def _parse_action_with_coords(response: str, available: list[int]) -> tuple[int, int, int]:
    """Extract action number and optional x,y coordinates from LLM response."""
    action = available[0]
    x, y = 32, 32  # default center

    try:
        # Try to find JSON
        m = re.search(r'\{[^}]*\}', response)
        if m:
            import json
            data = json.loads(m.group())
            action = int(data.get("action", available[0]))
            if action not in available:
                action = available[0]
            x = int(data.get("x", 32))
            y = int(data.get("y", 32))
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            return action, x, y
    except Exception:
        pass

    # Fallback
    action = _parse_action(response, available)
    return action, x, y


def _summarize_trajectory(trajectory: list[dict], llm_call_fn, model: str) -> str:
    """LLM summarizes what was learned from a play session."""
    if not trajectory:
        return "No actions taken."

    key_events = []
    for t in trajectory:
        if "LEVEL" in t.get("result", "") or t["step"] < 3 or t["step"] == len(trajectory) - 1:
            key_events.append(f"Step {t['step']}: action={t['action']} → {t['result']}")

    prompt = (
        f"You just played an abstract puzzle game for {len(trajectory)} steps.\n"
        f"Levels completed: {trajectory[-1]['levels']}\n\n"
        f"Key events:\n" + "\n".join(key_events[:10]) + "\n\n"
        "Summarize in 2-3 sentences: what rules did you discover? "
        "What strategy worked or didn't work?"
    )
    try:
        return llm_call_fn(model=model, user=prompt, max_tokens=150)
    except Exception:
        return f"Played {len(trajectory)} steps, completed {trajectory[-1]['levels']} levels."


# ---- EvoPool integration ----

def load_arc_agi3_tasks(game_ids: list[str] | None = None) -> list[dict]:
    """
    Load ARC-AGI-3 environments as EvoPool tasks.
    Each environment = 1 task with multi-round interaction.
    """
    arcade = arc_agi.Arcade()
    envs = arcade.get_environments()

    if game_ids is None:
        game_ids = [e.game_id.split('-')[0] for e in envs]

    tasks = []
    for gid in game_ids:
        matching = [e for e in envs if e.game_id.startswith(gid)]
        if not matching:
            continue
        e_info = matching[0]
        tasks.append({
            "id": f"arc3_{gid}",
            "type": "interactive_reasoning",
            "domain": "arc_agi3",
            "game_id": gid,
            "full_game_id": e_info.game_id,
            "title": e_info.title,
            "prompt": f"Play the abstract puzzle game '{e_info.title}'. "
                      f"Explore the environment, discover rules, and complete as many levels as possible.",
        })
    return tasks


class ArcAGI3Evaluator:
    """
    Evaluator for ARC-AGI-3 environments.

    Unlike math/code evaluators that check a single response,
    this evaluator runs agents interactively in the environment.
    Score = levels_completed / total_levels.
    """

    def __init__(self, max_steps_per_play: int = 150, warmup_plays: int = 1):
        self.max_steps = max_steps_per_play
        self.warmup_plays = warmup_plays
        self._arcade = arc_agi.Arcade()

    def __call__(self, task: dict, responses: dict[str, dict]) -> dict:
        """
        For ARC-AGI-3, 'responses' contains agent play results.
        Each response dict has 'score' and 'levels_completed' from agent_play().
        """
        scores = {}
        for agent_id, resp in responses.items():
            scores[agent_id] = resp.get("score", 0.0)
        scores["team_score"] = max(scores.values()) if scores else 0.0
        return scores

    def make_env(self, game_id: str):
        """Create a fresh environment instance."""
        return self._arcade.make(game_id)
