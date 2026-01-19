# leg_test/kinematics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PinToRobotIndexMap:
    """
    Maps pin q indices -> robot joints.
    Adjust once to match your URDF / closed-loop IK output.

    Default assumes a 6D pin configuration for the leg-ish chain:
      pin indices: 0,1,2,3,4,5
      motors: hipz(1), hipx(2), hipy(4), knee(3), ankle pair (5,6) from indices 4,5
    """
    hipz: int = 0
    hipx: int = 1
    hipy: int = 2
    knee: int = 3
    ankle_pitch: int = 4   # one pin ankle DOF
    ankle_roll: int = 5    # the other pin ankle DOF