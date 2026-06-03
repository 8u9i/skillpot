from skillopt import Edit, SkillOpt, SkillOptConfig, apply_edits, cosine_decay, insert_protected_section


def test_cosine_decay_reaches_start_and_end():
    assert cosine_decay(4, 2, 1, 4) == 4
    assert cosine_decay(4, 2, 4, 4) == 2


def test_apply_edits_preserves_protected_section():
    skill = "A\n<!-- SLOW_UPDATE_START -->\nprotected\n<!-- SLOW_UPDATE_END -->\nB"
    edited = apply_edits(skill, [Edit(op="replace", target="A", content="AA"), Edit(op="append", content="C")])

    assert "AA" in edited
    assert "protected" in edited
    assert edited.index("protected") < edited.index("B")
    assert edited.rstrip().endswith("C")


def test_insert_protected_section_replaces_existing_section():
    skill = insert_protected_section("base", "old")
    updated = insert_protected_section(skill, "new")

    assert "new" in updated
    assert "old" not in updated
    assert updated.count("<!-- SLOW_UPDATE_START -->") == 1


class StaticOptimizer:
    def __init__(self, edit):
        self.edit = edit

    def reflect_failures(self, trajectories, skill, rejected_buffer, budget):
        return [self.edit] if trajectories else []

    def reflect_successes(self, trajectories, skill, budget):
        return []

    def merge_edits(self, failure_edits, success_edits, skill):
        return list(failure_edits) + list(success_edits)

    def rank_and_select(self, merged_edits, skill, budget):
        return list(merged_edits[:budget])

    def slow_update(self, previous_skill, current_skill, previous_meta_skill, comparison):
        return "slow guidance"

    def meta_update(self, previous_skill, current_skill, previous_meta_skill, comparison):
        return "meta guidance"


def test_validation_gate_accepts_only_strict_improvement():
    def harness(task, skill):
        return {}, 1.0 if "fix" in skill else 0.0

    runner = SkillOpt(
        harness=harness,
        optimizer=StaticOptimizer(Edit(op="append", content="fix")),
        training_tasks=[1],
        selection_tasks=[1],
        test_tasks=[1],
        config=SkillOptConfig(epochs=1, rollout_batch_size=1, slow_update_samples=1),
    )

    state = runner.run("base")

    assert state.best_score == 1.0
    assert "fix" in state.best_skill
    assert "slow guidance" in state.best_skill


def test_validation_gate_rejects_equal_score_candidate():
    def harness(task, skill):
        return {}, 0.0

    runner = SkillOpt(
        harness=harness,
        optimizer=StaticOptimizer(Edit(op="append", content="unneeded")),
        training_tasks=[1],
        selection_tasks=[1],
        test_tasks=[1],
        config=SkillOptConfig(epochs=1, rollout_batch_size=1, slow_update_samples=1),
    )

    state = runner.run("base")

    assert "unneeded" not in state.best_skill
    assert len(state.rejected_buffer) == 1
    assert "slow guidance" in state.best_skill
