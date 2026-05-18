# LeRobot Humanoid Design

This repository contains the early-stage design work for the next LeRobot humanoid: URDF modeling, mechanical co-design studies, and optimal-control-based validation.

For the detailed methodology, equations, assumptions, and first results, see:
`docs/lerobot_humanoid_design_notes.pdf`

## Project Aim

The goal is to iteratively co-design a compact, maintainable humanoid platform for learning-based control and reproducible robotics research.
This repository currently focuses on the first milestone of that roadmap:

- Build a consistent baseline URDF model (`humanoid_v0`).
- Validate feasibility assumptions with optimal-control-based studies.
- Use optimization outputs to guide mechanical/CAD iterations.

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

## Co-Design Methodology

The project uses an iterative mechanical/control co-design loop:

1. Define a baseline robot model with explicit hardware assumptions (`humanoid_v0` mass, inertia, actuator limits, geometry).
2. Choose a parametric design space for the subsystem under study (for example hip axis orientations, upper-body joint placements).
3. Define benchmark tasks that represent the expected use cases.
4. Evaluate each design candidate by solving task-level problems:
   - Hip: locomotion OCPs (forward walk + side walk) solved with Sobec/Crocoddyl.
   - Upper body: closed-loop constrained IK + local refinement along motion trajectories under force/manipulability objectives.
5. Aggregate task costs into a scalar design objective (for hip: `0.66 * J_walk + 0.33 * J_side`).
6. Run an outer optimization loop (CMA-ES), keep best vectors, and feed results back to CAD/URDF updates.
7. Repeat the cycle as assumptions and constraints are refined.

This repository captures the first implemented iteration of this methodology (hip completed, upper body ongoing).

## Why OCP Helps Design Choices

An OCP solver is used here as a fast design critic, not just as a motion generator.
This is true even without a co-design optimizer: a single OCP run is already a quick integrity check for a given robot design.

Without any optimization loop, OCP helps answer:
- Is the design dynamically feasible for a target task?
- Are actuator limits respected with margin, or constantly saturated?
- Does the solver converge robustly, or only with fragile behavior?
- Which joints/axes are the bottlenecks in torque or tracking?

Because this check takes only a few seconds, it is practical during day-to-day design iteration (URDF/CAD updates) before committing to hardware changes.

For each candidate geometry:
1. Inject the design vector into the model (joint placements / axis orientations).
2. Solve the same benchmark task OCP with the same settings.
3. Read objective terms and constraints (torque-limit penalties, tracking error, velocity target, solver convergence).
4. Keep candidates that are feasible and low-cost; reject those that only work with high penalties or unstable convergence.

In practice, when one solve takes only a few seconds, we can evaluate many candidates in a CMA-ES loop.
This turns "can this design move?" into a quantitative score that is fast enough for iterative mechanical co-design.

## Upper-Body Co-Design Status (WIP)

- The tooling supports evaluation of shoulder and elbow-related design vectors (`codesign/upper_body`).
- In practice, elbow-focused axis optimization currently gives usable behavior.
- Shoulder optimization is currently not reliably convergent, so upper-body results are considered exploratory.
- The current structure is intentional for development: it allows cost-function tuning and debugging separately from geometric modeling changes.

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
