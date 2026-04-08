# Backward-compatibility shim.
# The canonical module is now rl_agent.py  — import from there.
from rl_agent import *  # noqa: F401, F403
from rl_agent import (  # noqa: F401
    AgentSpec,
    PolicyWrapper,
    RLAgent,
    POLICY_ACTION_KEYS,
    SNAPSHOT_TO_POLICY_JOINT_IDX,
    JOINT_TORQUE_TERM_NAMES,
    _load_config,
    _extract_default_joint_pos_rad_from_cfg,
)
