# LeRobot Humanoid Design

This repository gathers the design work for the next humanoid robot developed within the LeRobot team.

It centralizes early-stage engineering and research efforts related to robot modeling, mechanical co-design, and validation through optimization and optimal control.

---

## Goals

The goal of this repository is to centralize all design-related artifacts for the new humanoid robot, including:

- Robot models (URDF)
- Mechanical co-design algorithms and experiments
- Verification and validation through optimal control
- Design reports and internal technical notes
- Methodology documentation for future extensions

This repository is intended to evolve alongside the robot design.

---
## Installation


Requirements

The required Python environment is described in environment.lock.

In addition, the following dependencies are required:

A fork of Crocoddyl: https://github.com/LudovicDeMatteis/crocoddyl/tree/topic/contact-6D-closed-loop

A fork of Sobec : wip

meshcat (for visualization)
---

## Repository structure

```text
.
├── urdf/
│   ├── humanoid_v0/
│   │   └── urdf/
│   │       └── robot.urdf
│   │
│   └── humanoids_loader.py
│
├── codesign/
│   ├── hip/
│   │   
│
├── docs/
│   ├── humanoid_design_notes.pdf
│   └── figures/
│
└── README.md

