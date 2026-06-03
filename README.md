# SKILLOPT

SKILLOPT is a small, dependency-free Python implementation of a skill optimization loop for AI agents.

It treats the target model as frozen and improves a deployable skill document through:

- deterministic harness scoring
- rollout batches over training tasks
- failure and success reflection
- small edit budgets with cosine decay
- strict validation-gated acceptance
- rejection memory
- protected slow-update guidance

The implementation lives in [`skillopt/`](skillopt/). See [`skillopt/README.md`](skillopt/README.md) for usage examples and the optimizer adapter contract.

The file `Adobe Express - file.png` is preserved from the original repository cleanup request.
