# SKILLOPT

Dependency-free Python implementation of the SKILLOPT loop from the pasted spec.

The framework keeps the target model and harness frozen. You supply:

- `Harness`: runs one task with a skill and returns `(trajectory, score)` where score is deterministic in `[0, 1]`.
- `Optimizer`: proposes, merges, ranks, and maintains slow/meta guidance.
- Disjoint `training_tasks`, `selection_tasks`, and `test_tasks`.

## Minimal Usage

```python
from skillopt import Edit, SkillOpt, SkillOptConfig


def harness(task, skill):
    passed = task["answer"] in skill
    return {"passed": passed}, 1.0 if passed else 0.0


class ToyOptimizer:
    def reflect_failures(self, trajectories, skill, rejected_buffer, budget):
        return [Edit(op="append", content=f"Remember: {t.task['answer']}") for t in trajectories]

    def reflect_successes(self, trajectories, skill, budget):
        return []

    def merge_edits(self, failure_edits, success_edits, skill):
        return list(failure_edits) + list(success_edits)

    def rank_and_select(self, merged_edits, skill, budget):
        return list(merged_edits[:budget])

    def slow_update(self, previous_skill, current_skill, previous_meta_skill, comparison):
        return "Keep durable, cross-task instructions here."

    def meta_update(self, previous_skill, current_skill, previous_meta_skill, comparison):
        return "Prefer small edits that fix repeated failures."


runner = SkillOpt(
    harness=harness,
    optimizer=ToyOptimizer(),
    training_tasks=[{"answer": "alpha"}, {"answer": "beta"}],
    selection_tasks=[{"answer": "alpha"}],
    test_tasks=[{"answer": "beta"}],
    config=SkillOptConfig(epochs=1, rollout_batch_size=2),
)

state = runner.run("Initial skill")
runner.export_best(state.best_skill, "best_skill.md")
print(runner.final_test_score(state.best_skill))
```

## LLM Optimizer Adapter

Use `PromptOptimizer` when your optimizer model is exposed as a callable:

```python
from skillopt import PromptOptimizer


def generate(prompt_name, payload):
    # Call your optimizer model here and return JSON text or a dict.
    ...


optimizer = PromptOptimizer(generate)
```

The adapter uses these prompt names from the spec:

- `analyst_error.md`
- `analyst_success.md`
- `merge_priority.md`
- `ranking.md`
- `slow_update.md`
- `meta_skill.md`

## Implemented Constraints

- Edit budget follows cosine decay from `L_start` to `L_end`.
- Candidate skills are accepted only on strict selection-score improvement.
- Rejected edits are retained and truncated to the configured buffer size.
- Step-level edits cannot target or write slow-update markers.
- The slow-update section is replaced only at epoch boundaries.
- Scores outside `[0, 1]` raise an error.
