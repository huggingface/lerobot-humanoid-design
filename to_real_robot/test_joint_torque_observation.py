from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import can  # type: ignore  # noqa: F401
except Exception:
    can_stub = types.ModuleType("can")

    class _StubBus:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _StubMessage:
        def __init__(self, *args, **kwargs) -> None:
            pass

    can_stub.interface = types.SimpleNamespace(Bus=_StubBus)
    can_stub.Bus = _StubBus
    can_stub.Message = _StubMessage
    sys.modules["can"] = can_stub

from RL_agent import AgentSpec as MainAgentSpec
from RL_agent import POLICY_BLOCK_ACTION_KEYS_12
from RL_agent import RLAgent as MainRLAgent
from RL_agent import _extract_onnx_joint_action_keys
from RL_agent import _default_action_keys
from RL_agent import _load_config as load_main_config
from RL_agent import infer_agent_spec as infer_main_agent_spec
from RL_agent_isolated import AgentSpec as IsolatedAgentSpec
from RL_agent_isolated import POLICY_ACTION_KEYS as ISOLATED_POLICY_ACTION_KEYS
from RL_agent_isolated import RLAgent as IsolatedRLAgent
from RL_agent_isolated import infer_agent_spec as infer_isolated_agent_spec
from bipedal_robot import BipedalRobotController
from leg_test.mit import MotorState
from root_constant import ANKLE_COUPLING_CALIBRATION_LEFT, ANKLE_COUPLING_CALIBRATION_RIGHT, MOTOR_IDS


class _DummyPolicy:
    expected_input_dim = None


class _DummyRobot:
    pass


class _CaptureRobot:
    def __init__(self) -> None:
        self.last_left: dict[str, float] | None = None
        self.last_right: dict[str, float] | None = None

    def set_action(self, *, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        self.last_left = dict(left)
        self.last_right = dict(right)
        return {}

    def get_combined_state_snapshot(self, *, include_joint_state: bool = True) -> dict[str, object]:
        _ = include_joint_state
        return {
            "joint_state_deg": [0.0] * 12,
            "post_reset_hold_active": False,
            "post_reset_hold_remaining_s": 0.0,
        }


def _make_snapshot(joint_torque_nm: np.ndarray, *, time_s: float) -> dict[str, object]:
    return {
        "time_s": float(time_s),
        "joint_state_deg": [0.0] * 12,
        "joint_velocity_rad_s": [0.0] * 12,
        "joint_torque_nm": np.asarray(joint_torque_nm, dtype=np.float32).reshape(12).tolist(),
        "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def _make_snapshot_full(
    *,
    joint_state_deg: np.ndarray,
    joint_velocity_rad_s: np.ndarray,
    joint_torque_nm: np.ndarray,
    time_s: float,
) -> dict[str, object]:
    return {
        "time_s": float(time_s),
        "joint_state_deg": np.asarray(joint_state_deg, dtype=np.float32).reshape(12).tolist(),
        "joint_velocity_rad_s": np.asarray(joint_velocity_rad_s, dtype=np.float32).reshape(12).tolist(),
        "joint_torque_nm": np.asarray(joint_torque_nm, dtype=np.float32).reshape(12).tolist(),
        "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


class JointTorqueObservationTests(unittest.TestCase):
    def assertDictFloatAllClose(self, actual: dict[str, float] | None, expected: dict[str, float]) -> None:
        self.assertIsNotNone(actual)
        if actual is None:
            return
        self.assertEqual(set(actual.keys()), set(expected.keys()))
        for key, value in expected.items():
            self.assertAlmostEqual(float(actual[key]), float(value), places=5)

    def test_ankle_velocity_and_torque_roundtrip_respects_motor_sign(self) -> None:
        robot = BipedalRobotController(bus_can0=object(), bus_can1=object())
        qd_joint = np.zeros(12, dtype=float)
        tau_joint = np.zeros(12, dtype=float)
        qd_joint[[4, 5, 10, 11]] = [1.25, -0.75, -1.5, 0.5]
        tau_joint[[4, 5, 10, 11]] = [0.9, -0.4, -1.1, 0.3]

        qd_raw, tau_raw = robot._joint_vel_tau_to_motor_raw(qd_joint, tau_joint)
        qd_back = robot.motor_velocity_to_joint_velocity(qd_raw, output_radians=False, nq=12)
        tau_back = robot.motor_torque_to_joint_torque(tau_raw, nq=12)

        np.testing.assert_allclose(qd_back[[4, 5, 10, 11]], qd_joint[[4, 5, 10, 11]], atol=1e-6)
        np.testing.assert_allclose(tau_back[[4, 5, 10, 11]], tau_joint[[4, 5, 10, 11]], atol=1e-6)

    def test_bipedal_snapshot_exposes_joint_torque(self) -> None:
        robot = BipedalRobotController(bus_can0=object(), bus_can1=object())
        tau_raw = {mid: float(mid) for mid in MOTOR_IDS}
        for mid in MOTOR_IDS:
            robot.state[mid] = MotorState(
                position_deg=0.0,
                velocity_deg_s=0.0,
                torque_nm=tau_raw[mid],
                temp_mos_c=0.0,
                stamp=1.0,
            )

        snapshot = robot.get_combined_state_snapshot(include_joint_state=True)
        joint_tau = np.asarray(snapshot["joint_torque_nm"], dtype=float)

        expected = np.zeros(12, dtype=float)
        expected[0] = tau_raw[1] / robot.motor_sign[1]
        expected[1] = tau_raw[2] / robot.motor_sign[2]
        expected[2] = tau_raw[3] / robot.motor_sign[3]
        expected[3] = tau_raw[4] / robot.motor_sign[4]
        expected[6] = tau_raw[7] / robot.motor_sign[7]
        expected[7] = tau_raw[8] / robot.motor_sign[8]
        expected[8] = tau_raw[9] / robot.motor_sign[9]
        expected[9] = tau_raw[10] / robot.motor_sign[10]

        t5_cal = tau_raw[5] / robot.motor_sign[5]
        t6_cal = tau_raw[6] / robot.motor_sign[6]
        t11_cal = tau_raw[11] / robot.motor_sign[11]
        t12_cal = tau_raw[12] / robot.motor_sign[12]
        sp_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["pitch"]["sign"])
        sr_l = float(ANKLE_COUPLING_CALIBRATION_LEFT["roll"]["sign"])
        sp_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["pitch"]["sign"])
        sr_r = float(ANKLE_COUPLING_CALIBRATION_RIGHT["roll"]["sign"])
        expected[4] = (t5_cal - t6_cal) / sp_l
        expected[5] = (t5_cal + t6_cal) / sr_l
        expected[10] = (t11_cal - t12_cal) / sp_r
        expected[11] = (t11_cal + t12_cal) / sr_r

        np.testing.assert_allclose(joint_tau, expected, atol=1e-6)

    def test_isolated_rl_agent_uses_current_policy_order_joint_torque(self) -> None:
        agent = IsolatedRLAgent(
            robot=_DummyRobot(),
            spec=IsolatedAgentSpec(
                action_keys=list(ISOLATED_POLICY_ACTION_KEYS),
                policy_terms=["joint_torque"],
            ),
            policy=_DummyPolicy(),
        )

        first_tau = np.arange(1, 13, dtype=np.float32)
        obs_first = agent._build_obs_now(_make_snapshot(first_tau, time_s=0.0))
        obs_second = agent._build_obs_now(_make_snapshot(np.zeros(12, dtype=np.float32), time_s=1.0))

        expected_policy_order = np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=np.float32)
        np.testing.assert_allclose(obs_first, expected_policy_order, atol=1e-6)
        np.testing.assert_allclose(obs_second, np.zeros(12, dtype=np.float32), atol=1e-6)

    def test_isolated_rl_agent_reorders_joint_pos_and_scales_current_joint_torque(self) -> None:
        agent = IsolatedRLAgent(
            robot=_DummyRobot(),
            spec=IsolatedAgentSpec(
                action_keys=list(ISOLATED_POLICY_ACTION_KEYS),
                policy_terms=["joint_pos", "joint_torque"],
                obs_term_scales={"joint_torque": 0.5},
            ),
            policy=_DummyPolicy(),
        )
        agent._default_joint_pos_rad = np.zeros(12, dtype=np.float32)

        q_deg = np.arange(1, 13, dtype=np.float32)
        tau = np.arange(201, 213, dtype=np.float32)

        obs_first = agent._build_obs_now(
            _make_snapshot_full(
                joint_state_deg=q_deg,
                joint_velocity_rad_s=np.zeros(12, dtype=np.float32),
                joint_torque_nm=tau,
                time_s=0.0,
            )
        )
        obs_second = agent._build_obs_now(
            _make_snapshot_full(
                joint_state_deg=np.zeros(12, dtype=np.float32),
                joint_velocity_rad_s=np.zeros(12, dtype=np.float32),
                joint_torque_nm=np.zeros(12, dtype=np.float32),
                time_s=1.0,
            )
        )

        expected_pos = np.deg2rad(np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=np.float32))
        expected_tau = 0.5 * np.array([207, 208, 209, 210, 211, 212, 201, 202, 203, 204, 205, 206], dtype=np.float32)

        np.testing.assert_allclose(obs_first[:12], expected_pos, atol=1e-6)
        np.testing.assert_allclose(obs_first[12:24], expected_tau, atol=1e-6)
        np.testing.assert_allclose(obs_second[:12], expected_pos, atol=1e-6)
        np.testing.assert_allclose(obs_second[12:24], np.zeros(12, dtype=np.float32), atol=1e-6)

    def test_isolated_rl_agent_actions_obs_uses_last_raw_policy_action(self) -> None:
        agent = IsolatedRLAgent(
            robot=_DummyRobot(),
            spec=IsolatedAgentSpec(
                action_keys=list(ISOLATED_POLICY_ACTION_KEYS),
                policy_terms=["actions"],
                action_scale=0.0,
            ),
            policy=_DummyPolicy(),
        )

        raw_action = np.linspace(-1.5, 1.5, 12, dtype=np.float32)
        agent._apply_action(raw_action)
        obs_zero_scale = agent._term_observation_vector({}, "actions")
        np.testing.assert_allclose(obs_zero_scale, raw_action, atol=1e-6)

        agent.spec.action_scale = 0.25
        agent._apply_action(raw_action)
        obs_quarter_scale = agent._term_observation_vector({}, "actions")
        np.testing.assert_allclose(obs_quarter_scale, raw_action, atol=1e-6)

        agent.spec.debug_zero_actions_obs = True
        obs_zeroed = agent._term_observation_vector({}, "actions")
        np.testing.assert_allclose(obs_zeroed, np.zeros_like(raw_action), atol=1e-6)
        np.testing.assert_allclose(agent._last_policy_action, raw_action, atol=1e-6)

    def test_isolated_rl_agent_apply_action_maps_policy_right_then_left_to_robot_sides(self) -> None:
        robot = _CaptureRobot()
        agent = IsolatedRLAgent(
            robot=robot,
            spec=IsolatedAgentSpec(
                action_keys=list(ISOLATED_POLICY_ACTION_KEYS),
                action_scale=1.0,
                action_scales_rad=[1.0] * 12,
                encoder_bias_rad=[0.0] * 12,
            ),
            policy=_DummyPolicy(),
        )
        agent._default_joint_pos_rad = np.zeros(12, dtype=np.float32)

        action_rad = np.deg2rad(np.arange(1, 13, dtype=np.float32))
        agent._apply_action(action_rad)

        self.assertDictFloatAllClose(
            robot.last_right,
            {
                "hipz": 1.0,
                "hipx": 2.0,
                "hipy": 3.0,
                "knee": 4.0,
                "ankle_pitch": 5.0,
                "ankle_roll": 6.0,
            },
        )
        self.assertDictFloatAllClose(
            robot.last_left,
            {
                "hipz": 7.0,
                "hipx": 8.0,
                "hipy": 9.0,
                "knee": 10.0,
                "ankle_pitch": 11.0,
                "ankle_roll": 12.0,
            },
        )

    def test_main_rl_agent_uses_delayed_joint_torque(self) -> None:
        agent = MainRLAgent(
            robot=_DummyRobot(),
            spec=MainAgentSpec(
                action_keys=_default_action_keys(),
                policy_terms=["joint_torque"],
            ),
            policy=_DummyPolicy(),
        )

        first_tau = np.arange(1, 13, dtype=np.float32)
        obs_first = agent._build_obs_now(_make_snapshot(first_tau, time_s=0.0))
        obs_second = agent._build_obs_now(_make_snapshot(np.zeros(12, dtype=np.float32), time_s=1.0))

        np.testing.assert_allclose(obs_first, np.zeros(12, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs_second, first_tau, atol=1e-6)

    def test_main_rl_agent_policy_joint_order_reorders_joint_terms_and_scales_torque(self) -> None:
        agent = MainRLAgent(
            robot=_DummyRobot(),
            spec=MainAgentSpec(
                action_keys=list(POLICY_BLOCK_ACTION_KEYS_12),
                policy_terms=["joint_pos", "joint_vel", "joint_torque"],
                obs_term_scales={"joint_torque": 0.5},
                use_policy_joint_order=True,
            ),
            policy=_DummyPolicy(),
        )
        agent._default_joint_pos_rad = np.zeros(12, dtype=np.float32)

        q_deg = np.arange(1, 13, dtype=np.float32)
        qd = np.arange(101, 113, dtype=np.float32)
        tau = np.arange(201, 213, dtype=np.float32)

        obs_first = agent._build_obs_now(
            _make_snapshot_full(
                joint_state_deg=q_deg,
                joint_velocity_rad_s=qd,
                joint_torque_nm=tau,
                time_s=0.0,
            )
        )
        obs_second = agent._build_obs_now(
            _make_snapshot_full(
                joint_state_deg=np.zeros(12, dtype=np.float32),
                joint_velocity_rad_s=np.zeros(12, dtype=np.float32),
                joint_torque_nm=np.zeros(12, dtype=np.float32),
                time_s=1.0,
            )
        )

        expected_pos = np.deg2rad(np.array([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=np.float32))
        expected_pos[[2, 4, 8, 10]] *= -1.0
        expected_vel = np.array([107, 108, 109, 110, 111, 112, 101, 102, 103, 104, 105, 106], dtype=np.float32)
        expected_tau = 0.5 * np.array([207, 208, 209, 210, 211, 212, 201, 202, 203, 204, 205, 206], dtype=np.float32)

        np.testing.assert_allclose(obs_first[:12], expected_pos, atol=1e-6)
        np.testing.assert_allclose(obs_first[12:24], np.zeros(12, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs_first[24:36], np.zeros(12, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(obs_second[:12], expected_pos, atol=1e-6)
        np.testing.assert_allclose(obs_second[12:24], expected_vel, atol=1e-6)
        np.testing.assert_allclose(obs_second[24:36], expected_tau, atol=1e-6)

    def test_main_rl_agent_apply_action_maps_block_policy_order_to_robot_sides(self) -> None:
        robot = _CaptureRobot()
        agent = MainRLAgent(
            robot=robot,
            spec=MainAgentSpec(
                action_keys=list(POLICY_BLOCK_ACTION_KEYS_12),
                action_scale=1.0,
                action_scales_rad=[1.0] * 12,
                use_policy_joint_order=True,
            ),
            policy=_DummyPolicy(),
        )
        agent._default_joint_pos_rad = np.zeros(12, dtype=np.float32)

        action_rad = np.deg2rad(np.arange(1, 13, dtype=np.float32))
        agent._apply_action(action_rad)

        self.assertDictFloatAllClose(
            robot.last_right,
            {
                "hipz": 1.0,
                "hipx": 2.0,
                "hipy": 3.0,
                "knee": 4.0,
                "ankle_pitch": 5.0,
                "ankle_roll": 6.0,
            },
        )
        self.assertDictFloatAllClose(
            robot.last_left,
            {
                "hipz": 7.0,
                "hipx": 8.0,
                "hipy": 9.0,
                "knee": 10.0,
                "ankle_pitch": 11.0,
                "ankle_roll": 12.0,
            },
        )

    def test_main_infer_agent_spec_canonicalizes_torque_obs_yaml_order(self) -> None:
        cfg = load_main_config(Path("RL_policy/less_noice_high_gain_torque_obs/config.yaml"))
        spec = infer_main_agent_spec(cfg)
        self.assertEqual(
            spec.policy_terms,
            ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "joint_torques"],
        )

    def test_isolated_infer_agent_spec_canonicalizes_torque_obs_yaml_order(self) -> None:
        cfg = load_main_config(Path("RL_policy/less_noice_high_gain_torque_obs/config.yaml"))
        spec = infer_isolated_agent_spec(cfg)
        self.assertEqual(
            spec.policy_terms,
            ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "joint_torques"],
        )

    def test_extract_onnx_joint_action_keys_detects_block_policy_order(self) -> None:
        policy_path = Path("RL_policy/less_noice_high_gain_torque_obs/policy.onnx")
        if not policy_path.exists():
            self.skipTest(f"missing test asset: {policy_path}")
        self.assertEqual(_extract_onnx_joint_action_keys(policy_path), list(POLICY_BLOCK_ACTION_KEYS_12))


if __name__ == "__main__":
    unittest.main()
