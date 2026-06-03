from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence


SLOW_START = "<!-- SLOW_UPDATE_START -->"
SLOW_END = "<!-- SLOW_UPDATE_END -->"


@dataclass(frozen=True)
class Trajectory:
    task: Any
    trace: Any
    score: float


@dataclass(frozen=True)
class Edit:
    op: str
    content: str = ""
    target: str = ""


@dataclass
class RejectedEdit:
    edits: list[Edit]
    score_drop: float
    failures: str


@dataclass(frozen=True)
class SkillOptConfig:
    epochs: int = 4
    rollout_batch_size: int = 40
    reflection_minibatch_size: int = 8
    edit_budget_start: int = 4
    edit_budget_end: int = 2
    rejected_buffer_size: int = 20
    slow_update_samples: int = 20
    steps_per_epoch: int | None = None
    seed: int = 0


@dataclass
class SkillOptState:
    current_skill: str
    best_skill: str
    best_score: float
    rejected_buffer: list[RejectedEdit] = field(default_factory=list)
    meta_skill: str = ""
    slow_field: str = ""


@dataclass
class LongitudinalComparison:
    regressions: list[tuple[Any, float, float]]
    persistent_failures: list[tuple[Any, float, float]]
    improvements: list[tuple[Any, float, float]]
    stable_successes: list[tuple[Any, float, float]]


class Harness(Protocol):
    def __call__(self, task: Any, skill: str) -> tuple[Any, float]:
        """Run the frozen target model on one task with a skill."""


class Optimizer(Protocol):
    def reflect_failures(
        self,
        trajectories: Sequence[Trajectory],
        skill: str,
        rejected_buffer: Sequence[RejectedEdit],
        budget: int,
    ) -> list[Edit]:
        ...

    def reflect_successes(
        self,
        trajectories: Sequence[Trajectory],
        skill: str,
        budget: int,
    ) -> list[Edit]:
        ...

    def merge_edits(
        self,
        failure_edits: Sequence[Edit],
        success_edits: Sequence[Edit],
        skill: str,
    ) -> list[Edit]:
        ...

    def rank_and_select(
        self,
        merged_edits: Sequence[Edit],
        skill: str,
        budget: int,
    ) -> list[Edit]:
        ...

    def slow_update(
        self,
        previous_skill: str,
        current_skill: str,
        previous_meta_skill: str,
        comparison: LongitudinalComparison,
    ) -> str:
        ...

    def meta_update(
        self,
        previous_skill: str,
        current_skill: str,
        previous_meta_skill: str,
        comparison: LongitudinalComparison,
    ) -> str:
        ...


class PromptOptimizer:
    """Adapter for an LLM-style optimizer callable.

    The callable receives a prompt name and JSON-serializable payload, and should
    return either a JSON string or a Python dict matching the prompt contract.
    """

    def __init__(self, generate: Callable[[str, dict[str, Any]], str | dict[str, Any]]):
        self.generate = generate

    def reflect_failures(
        self,
        trajectories: Sequence[Trajectory],
        skill: str,
        rejected_buffer: Sequence[RejectedEdit],
        budget: int,
    ) -> list[Edit]:
        payload = {
            "failed_trajectories": [_trajectory_to_json(t) for t in trajectories],
            "current_skill": skill,
            "rejected_buffer": [_rejected_to_json(r) for r in rejected_buffer],
            "budget": budget,
        }
        return _edits_from_response(self.generate("analyst_error.md", payload))

    def reflect_successes(
        self,
        trajectories: Sequence[Trajectory],
        skill: str,
        budget: int,
    ) -> list[Edit]:
        payload = {
            "successful_trajectories": [_trajectory_to_json(t) for t in trajectories],
            "current_skill": skill,
            "budget": budget,
        }
        return _edits_from_response(self.generate("analyst_success.md", payload))

    def merge_edits(
        self,
        failure_edits: Sequence[Edit],
        success_edits: Sequence[Edit],
        skill: str,
    ) -> list[Edit]:
        payload = {
            "failure_edits": [_edit_to_json(e) for e in failure_edits],
            "success_edits": [_edit_to_json(e) for e in success_edits],
            "current_skill": skill,
        }
        return _edits_from_response(self.generate("merge_priority.md", payload))

    def rank_and_select(
        self,
        merged_edits: Sequence[Edit],
        skill: str,
        budget: int,
    ) -> list[Edit]:
        payload = {
            "edits": [_edit_to_json(e) for e in merged_edits],
            "current_skill": skill,
            "budget": budget,
        }
        response = _ensure_dict(self.generate("ranking.md", payload))
        indices = response.get("selected_indices", [])
        return [merged_edits[i] for i in indices[:budget] if isinstance(i, int) and 0 <= i < len(merged_edits)]

    def slow_update(
        self,
        previous_skill: str,
        current_skill: str,
        previous_meta_skill: str,
        comparison: LongitudinalComparison,
    ) -> str:
        response = _ensure_dict(
            self.generate(
                "slow_update.md",
                {
                    "previous_skill": previous_skill,
                    "current_skill": current_skill,
                    "previous_meta_skill": previous_meta_skill,
                    "comparison": _comparison_to_json(comparison),
                },
            )
        )
        return str(response.get("slow_update_content", "")).strip()

    def meta_update(
        self,
        previous_skill: str,
        current_skill: str,
        previous_meta_skill: str,
        comparison: LongitudinalComparison,
    ) -> str:
        response = _ensure_dict(
            self.generate(
                "meta_skill.md",
                {
                    "previous_skill": previous_skill,
                    "current_skill": current_skill,
                    "previous_meta_skill": previous_meta_skill,
                    "comparison": _comparison_to_json(comparison),
                },
            )
        )
        return str(response.get("meta_skill_content", "")).strip()


class SkillOpt:
    def __init__(
        self,
        harness: Harness,
        optimizer: Optimizer,
        training_tasks: Sequence[Any],
        selection_tasks: Sequence[Any],
        test_tasks: Sequence[Any],
        config: SkillOptConfig | None = None,
    ):
        self.harness = harness
        self.optimizer = optimizer
        self.training_tasks = list(training_tasks)
        self.selection_tasks = list(selection_tasks)
        self.test_tasks = list(test_tasks)
        self.config = config or SkillOptConfig()
        self.random = random.Random(self.config.seed)

    def evaluate(self, skill: str, task_set: Sequence[Any]) -> float:
        if not task_set:
            raise ValueError("task_set must not be empty")
        total = 0.0
        for task in task_set:
            _, score = self.harness(task, skill)
            total += _validate_score(score)
        return total / len(task_set)

    def rollout_batch(self, skill: str, tasks: Sequence[Any]) -> list[Trajectory]:
        trajectories = []
        for task in tasks:
            trace, score = self.harness(task, skill)
            trajectories.append(Trajectory(task=task, trace=trace, score=_validate_score(score)))
        return trajectories

    def run(self, initial_skill: str) -> SkillOptState:
        best_score = self.evaluate(initial_skill, self.selection_tasks)
        state = SkillOptState(
            current_skill=initial_skill,
            best_skill=initial_skill,
            best_score=best_score,
        )

        for epoch in range(1, self.config.epochs + 1):
            budget = cosine_decay(
                self.config.edit_budget_start,
                self.config.edit_budget_end,
                epoch,
                self.config.epochs,
            )
            epoch_start_skill = state.current_skill
            epoch_tasks = self._epoch_batches()

            for batch in epoch_tasks:
                trajectories = self.rollout_batch(state.current_skill, batch)
                failures = [t for t in trajectories if t.score < 1.0]
                successes = [t for t in trajectories if t.score == 1.0]

                failure_edits = []
                for minibatch in chunks(failures, self.config.reflection_minibatch_size):
                    failure_edits.extend(
                        self.optimizer.reflect_failures(minibatch, state.current_skill, state.rejected_buffer, budget)
                    )

                success_edits = []
                for minibatch in chunks(successes, self.config.reflection_minibatch_size):
                    success_edits.extend(self.optimizer.reflect_successes(minibatch, state.current_skill, budget))

                merged = self.optimizer.merge_edits(failure_edits, success_edits, state.current_skill)
                selected_edits = self.optimizer.rank_and_select(merged, state.current_skill, budget)
                selected_edits = selected_edits[:budget]
                if not selected_edits:
                    continue

                candidate = apply_edits(state.current_skill, selected_edits)
                candidate_score = self.evaluate(candidate, self.selection_tasks)

                if candidate_score > state.best_score:
                    state.current_skill = candidate
                    state.best_skill = candidate
                    state.best_score = candidate_score
                    state.rejected_buffer = state.rejected_buffer[-self.config.rejected_buffer_size :]
                else:
                    state.rejected_buffer.append(
                        RejectedEdit(
                            edits=list(selected_edits),
                            score_drop=state.best_score - candidate_score,
                            failures=summarize_failures(failures),
                        )
                    )
                    state.rejected_buffer = state.rejected_buffer[-self.config.rejected_buffer_size :]

            comparison = self._compare_skills(epoch_start_skill, state.current_skill)
            state.slow_field = self.optimizer.slow_update(
                epoch_start_skill,
                state.current_skill,
                state.meta_skill,
                comparison,
            )
            state.current_skill = insert_protected_section(state.current_skill, state.slow_field)
            state.best_skill = insert_protected_section(state.best_skill, state.slow_field)
            state.meta_skill = self.optimizer.meta_update(
                epoch_start_skill,
                state.current_skill,
                state.meta_skill,
                comparison,
            )

        return state

    def final_test_score(self, skill: str) -> float:
        return self.evaluate(skill, self.test_tasks)

    def export_best(self, skill: str, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(skill, encoding="utf-8")
        return destination

    def _epoch_batches(self) -> list[list[Any]]:
        tasks = list(self.training_tasks)
        self.random.shuffle(tasks)
        batches = list(chunks(tasks, self.config.rollout_batch_size))
        if self.config.steps_per_epoch is not None:
            return batches[: self.config.steps_per_epoch]
        return batches

    def _compare_skills(self, previous_skill: str, current_skill: str) -> LongitudinalComparison:
        sample_size = min(self.config.slow_update_samples, len(self.training_tasks))
        sample = self.random.sample(self.training_tasks, sample_size) if sample_size else []

        regressions = []
        persistent_failures = []
        improvements = []
        stable_successes = []

        for task in sample:
            _, previous_score = self.harness(task, previous_skill)
            _, current_score = self.harness(task, current_skill)
            previous_score = _validate_score(previous_score)
            current_score = _validate_score(current_score)

            item = (task, previous_score, current_score)
            if current_score < previous_score:
                regressions.append(item)
            elif previous_score < 1.0 and current_score < 1.0:
                persistent_failures.append(item)
            elif current_score > previous_score:
                improvements.append(item)
            elif previous_score == 1.0 and current_score == 1.0:
                stable_successes.append(item)

        return LongitudinalComparison(
            regressions=regressions,
            persistent_failures=persistent_failures,
            improvements=improvements,
            stable_successes=stable_successes,
        )


def cosine_decay(start: int, end: int, epoch: int, epochs: int) -> int:
    if epochs <= 1:
        return int(round(end))
    progress = (epoch - 1) / (epochs - 1)
    value = end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
    return max(0, int(round(value)))


def chunks(items: Sequence[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def apply_edits(skill: str, edits: Sequence[Edit]) -> str:
    editable_prefix, protected, editable_suffix = _split_protected_section(skill)

    for edit in edits:
        if not _edit_allowed(edit):
            continue
        if edit.op == "append":
            editable_suffix = editable_suffix.rstrip() + "\n" + edit.content.strip() + "\n"
        elif edit.op in {"insert_after", "replace", "delete"}:
            editable_prefix, editable_suffix = _apply_targeted_edit(editable_prefix, editable_suffix, edit)
        else:
            raise ValueError(f"unsupported or unmatchable edit: {edit}")

    if protected:
        return editable_prefix + protected + editable_suffix
    return editable_prefix + editable_suffix


def insert_protected_section(skill: str, slow_field: str) -> str:
    protected = f"{SLOW_START}\n{slow_field.strip()}\n{SLOW_END}"
    prefix, existing, suffix = _split_protected_section(skill)
    if existing:
        return (prefix.rstrip() + "\n\n" + protected + "\n" + suffix.lstrip()).strip() + "\n"
    return skill.rstrip() + "\n\n" + protected + "\n"


def summarize_failures(failures: Sequence[Trajectory]) -> str:
    if not failures:
        return "No failed trajectories in the sampled batch."
    summaries = []
    for failure in failures[:10]:
        summaries.append(f"task={failure.task!r}, score={failure.score:.3f}, trace={failure.trace!r}")
    if len(failures) > 10:
        summaries.append(f"... {len(failures) - 10} more failures")
    return "\n".join(summaries)


def _apply_targeted_edit(prefix: str, suffix: str, edit: Edit) -> tuple[str, str]:
    if edit.target in prefix:
        return _apply_edit_to_text(prefix, edit), suffix
    if edit.target in suffix:
        return prefix, _apply_edit_to_text(suffix, edit)
    raise ValueError(f"unsupported or unmatchable edit: {edit}")


def _apply_edit_to_text(text: str, edit: Edit) -> str:
    if edit.op == "insert_after":
        return text.replace(edit.target, edit.target + "\n" + edit.content.strip(), 1)
    if edit.op == "replace":
        return text.replace(edit.target, edit.content, 1)
    if edit.op == "delete":
        return text.replace(edit.target, "", 1)
    raise ValueError(f"unsupported or unmatchable edit: {edit}")


def _split_protected_section(skill: str) -> tuple[str, str, str]:
    start = skill.find(SLOW_START)
    end = skill.find(SLOW_END)
    if start == -1 and end == -1:
        return skill, "", ""
    if start == -1 or end == -1 or end < start:
        raise ValueError("malformed protected slow update section")
    end += len(SLOW_END)
    return skill[:start], skill[start:end], skill[end:]


def _edit_allowed(edit: Edit) -> bool:
    text = f"{edit.target}\n{edit.content}"
    return SLOW_START not in text and SLOW_END not in text


def _validate_score(score: float) -> float:
    score = float(score)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    return score


def _ensure_dict(response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    return json.loads(response)


def _edits_from_response(response: str | dict[str, Any]) -> list[Edit]:
    data = _ensure_dict(response)
    edits = []
    for raw in data.get("edits", []):
        edits.append(
            Edit(
                op=str(raw.get("op", "")),
                target=str(raw.get("target", "")),
                content=str(raw.get("content", "")),
            )
        )
    return edits


def _edit_to_json(edit: Edit) -> dict[str, str]:
    return {"op": edit.op, "target": edit.target, "content": edit.content}


def _trajectory_to_json(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "task": repr(trajectory.task),
        "trace": repr(trajectory.trace),
        "score": trajectory.score,
    }


def _rejected_to_json(rejected: RejectedEdit) -> dict[str, Any]:
    return {
        "edits": [_edit_to_json(edit) for edit in rejected.edits],
        "score_drop": rejected.score_drop,
        "failures": rejected.failures,
    }


def _comparison_to_json(comparison: LongitudinalComparison) -> dict[str, Any]:
    return {
        "regressions": [_comparison_item_to_json(i) for i in comparison.regressions],
        "persistent_failures": [_comparison_item_to_json(i) for i in comparison.persistent_failures],
        "improvements": [_comparison_item_to_json(i) for i in comparison.improvements],
        "stable_successes": [_comparison_item_to_json(i) for i in comparison.stable_successes],
    }


def _comparison_item_to_json(item: tuple[Any, float, float]) -> dict[str, Any]:
    task, previous_score, current_score = item
    return {
        "task": repr(task),
        "previous_score": previous_score,
        "current_score": current_score,
    }
