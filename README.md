# LeRobot Humanoid Design

This repository contains the early-stage design work for the next LeRobot humanoid: URDF modeling, mechanical co-design studies, and optimal-control-based validation.

For the detailed methodology, equations, assumptions, and first results, see:
`docs/lerobot_humanoid_design_notes.pdf`

## Scope

- URDF assets for baseline lower-body and upper-body studies.
- Hip co-design optimization loop (CMA-ES + OCP evaluation).
- Experiment scripts for walking, side-walking, and hip-axis sensitivity.
- Upper-body co-design tooling and evaluators.

## Current Status

- `humanoid_v0` baseline model is available and used in optimization/experiments.
- First hip co-design pipeline is operational.
- Bipedal-platform simulation studies are operational.
- Upper-body optimization is still WIP.

## Design Process (Summary)

The current process follows the roadmap documented in the design note:

1. Build a consistent baseline model (`humanoid_v0`) with actuator assumptions, mass/inertia estimates, and torque limits.
2. Parameterize hip geometry (axis arrangement) and sample candidate vectors.
3. Evaluate each candidate with OCPs (forward walking + side walking).
4. Aggregate the objective with a weighted scalar cost (`0.66 * J_walk + 0.33 * J_side`).
5. Iterate between model parameters, optimization outputs, and CAD updates.

The same co-design philosophy is now being extended to the upper body, but this part is not finalized yet.

## Repository Layout

- `urdf/`: robot models and loading utilities.
- `codesign/hip/`: hip optimization and vector evaluation scripts.
- `codesign/upper_body/`: upper-body evaluators and optimization scripts.
- `experiments/`: standalone experiment scripts and experiment-local parameter variants.
- `docs/`: technical notes and project documentation.

## Installation

The Python environment is described in `environement.lock`.

Additional dependencies used by the current scripts:

- Crocoddyl fork: `https://github.com/LudovicDeMatteis/crocoddyl/tree/topic/contact-6D-closed-loop`
- Sobec fork/work-in-progress (project-specific setup)
- `meshcat` for visualization
