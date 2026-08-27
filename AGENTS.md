# AGENT.md

## Remote Reference

- Remote repository: <https://github.com/Virgileboat/lerobot-humanoid-design>
- Default branch: <https://github.com/Virgileboat/lerobot-humanoid-design/tree/main>
- Current tracked upstream branch: `origin/main`

## Full Project Context

This repository is one part of the LeRobot humanoid stack:

1. `lerobot-humanoid-design`: co-design and feasibility studies (this repo)
2. `lerobot-humanoid-hardware`: build docs/BOM/assembly/electronics
3. `lerobot-humanoid-model`: shared MJCF/URDF model assets
4. `lerobot-humanoid-runtime`: simulation + real robot runtime and calibration
5. `lerobot-humanoid-identification`: offline replay + CMA-ES identification

Any change here should preserve compatibility with the other four repositories.

## Mission Of This Repo

Develop and validate mechanical/control design choices before hardware commits:

- baseline URDF maintenance
- hip/upper-body co-design experiments
- OCP-based feasibility scoring

## Key Interfaces

- Upstream constraints:
  - hardware feasibility and manufacturability from `lerobot-humanoid-hardware`
- Downstream consumers:
  - model packaging in `lerobot-humanoid-model`
  - runtime experiments in `lerobot-humanoid-runtime`
  - identification assumptions in `lerobot-humanoid-identification`

## Critical Paths In This Repo

- `urdf/`: robot definitions and loaders
- `codesign/hip/`: hip optimization loop and vector evaluation
- `codesign/upper_body/`: upper-body evaluators/optimizers
- `experiments/`: experiment scripts and param variants
- `docs/lerobot_humanoid_design_notes.pdf`: methodology and assumptions

## Working Rules

1. Keep optimization and evaluation scripts reproducible (same CLI should still run).
2. Keep model assumptions explicit when changing geometry/mass/inertia/limits.
3. Preserve compatibility of URDF assets with downstream model/runtime tooling.
4. Do not silently remove benchmark tasks used for historical comparison.
5. If a design vector format changes, update the relevant docs/scripts in the same change.

## Validation Before Merge

- Run at least one representative hip evaluation path and one experiment script.
- Confirm URDF loaders still resolve expected files.
- Document behavioral deltas (why this design is better/worse and under which task).
