# leg_test/leg_robot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import time
import threading
import numpy as np
import can

from .config import (
    LEG_MOTORS, LEG_MOTOR_IDS, DEFAULT_GAINS,
    CAN_CMD_CLEAR_FAULT, CAN_CMD_ZERO, CAN_CMD_ENABLE, CAN_CMD_DISABLE,
    JOINT_LIMITS_DEG, COMMAND_MARGIN_DEG, STATE_MARGIN_DEG,
)
from .mit import MotorState, decode_state_frame, pack_mit_command
import pinocchio as pin
from toolbox_parallel_robots.inverse_kinematics import closedLoopInverseKinematicsProximal
from .kinematics import PinToRobotIndexMap

@dataclass
class JointGains:
    kp: float
    kd: float


class SafetyError(RuntimeError):
    pass


class LegRobot:
    def __init__(
        self,
        *,
        interface: str = "socketcan",
        channel: str = "can0",
        recv_timeout_s: float = 0.02,
    ):
        self.bus = can.interface.Bus(interface=interface, channel=channel)
        self.recv_timeout_s = float(recv_timeout_s)

        self.state: Dict[int, MotorState] = {mid: MotorState() for mid in LEG_MOTOR_IDS}
        self._state_lock = threading.Lock()

        self.gains: Dict[int, JointGains] = {
            mid: JointGains(*DEFAULT_GAINS.get(mid, (5.0, 0.5)))
            for mid in LEG_MOTOR_IDS
        }

        # -------------------------
        # SAFETY / ESTOP
        # -------------------------
        self.estop: bool = False
        self.estop_reason: str = ""
        self.estop_stamp: float = 0.0

        # Viz
        self._viz = None
        self._viz_thread: Optional[threading.Thread] = None
        self._viz_stop = False
        self._viz_hz = 30.0



        # -------------------------
        # Kinematics / IK
        # -------------------------
        self.model = None
        self.constraint_models = None
        self.actuation_model = None
        self.visual_model = None
        self.collision_model = None

        self._pin_map = PinToRobotIndexMap()
        self._q_pin = None  # last pin configuration (np.ndarray)



    def attach_kinematics(
        self,
        *,
        model,
        constraint_models=None,
        actuation_model=None,
        visual_model=None,
        collision_model=None,
        pin_map: Optional[PinToRobotIndexMap] = None,
    ) -> None:
        """
        Attach pinocchio model & related structures so the robot can run IK.
        """
        self.model = model
        self.constraint_models = constraint_models
        self.actuation_model = actuation_model
        self.visual_model = visual_model
        self.collision_model = collision_model

        if pin_map is not None:
            self._pin_map = pin_map

        # initialize q_pin
        try:
            self._q_pin = pin.neutral(self.model).copy()
        except Exception:
            self._q_pin = None



    def pin_to_robot_deg(self, q_pin: np.ndarray) -> Dict[int, float]:
        """
        Convert q_pin (rad) to motor target positions (deg) keyed by motor_id.

        NOTE: ankle mixing is convention-dependent. The default below is:
          ankle1 =  +pitch - roll
          ankle2 =  +pitch + roll
        If your ankles move the wrong way, flip signs here (single place).
        """
        q_pin = np.asarray(q_pin, dtype=float)

        m = self._pin_map
        hipz = float(q_pin[m.hipz])
        hipx = float(q_pin[m.hipx])
        hipy = (float(q_pin[m.hipy])-2*np.pi)%(2*np.pi)


        knee = (float(q_pin[m.knee])- 2 *np.pi)%(2*np.pi)

        ankle_pitch = float(q_pin[m.ankle_pitch])
        ankle_roll  = float(q_pin[m.ankle_roll])

        # --- ankle motor mixing (EDIT HERE if needed) ---
        ankle1 = ((ankle_pitch - ankle_roll)-2*np.pi)%2*np.pi
        ankle2 = ((ankle_pitch + ankle_roll)+2*np.pi)%2*np.pi

        return {
            1: np.degrees(hipz),
            2: np.degrees(hipx),
            4: np.degrees(hipy),
            3: np.degrees(knee),
            5: np.degrees(ankle1),
            6: np.degrees(ankle2),
        }

    def ik_foot_translation(
        self,
        target_pos: np.ndarray,
        *,
        q_prec: Optional[np.ndarray] = None,
        foot_frame_name: str = "foot",
    ) -> np.ndarray:
        """
        Closed-loop IK: moves the foot frame translation to target_pos (3,)
        Returns q_pin (np.ndarray).

        Uses toolbox_parallel_robots.closedLoopInverseKinematicsProximal exactly like your snippet.
        """
        if self.model is None:
            raise RuntimeError("No pinocchio model attached. Call attach_kinematics() first.")

        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        if q_prec is None:
            q_prec = self._q_pin
        if q_prec is None:
            q_prec = pin.neutral(self.model)

        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q_prec)

        target_M = pin.SE3(np.eye(3), target_pos)

        q = closedLoopInverseKinematicsProximal(
            self.model,
            data,
            [],              # constraint models (you used empty)
            [],              # constraint data (you used empty)
            target_M,
            foot_frame_name,
            q_prec=q_prec.tolist(),
        )

        q = np.asarray(q, dtype=float)
        self._q_pin = q.copy()
        return q

    # -------------------------
    # Safety helpers
    # -------------------------
    def _limits_for(self, motor_id: int) -> Optional[Tuple[float, float]]:
        return JOINT_LIMITS_DEG.get(motor_id)

    def _check_state_in_bounds(self, motor_id: int, pos_deg: float) -> bool:
        lim = self._limits_for(motor_id)
        if lim is None:
            return True
        lo, hi = lim
        lo -= STATE_MARGIN_DEG
        hi += STATE_MARGIN_DEG
        return (lo <= pos_deg <= hi)

    def _clamp_command(self, motor_id: int, pos_deg: float) -> float:
        lim = self._limits_for(motor_id)
        if lim is None:
            return pos_deg
        lo, hi = lim
        lo += COMMAND_MARGIN_DEG
        hi -= COMMAND_MARGIN_DEG
        # if margins invert the range, fall back to raw limits
        if lo > hi:
            lo, hi = lim
        return float(np.clip(pos_deg, lo, hi))

    def _assert_can_send(self) -> None:
        if self.estop:
            raise SafetyError(f"E-STOP active: {self.estop_reason}")

    def clear_estop(self) -> None:
        """
        Re-arm the robot after an estop. Does NOT re-enable motors automatically.
        """
        self.estop = False
        self.estop_reason = ""
        self.estop_stamp = 0.0

    def emergency_stop(self, reason: str, motor_ids: Iterable[int] = LEG_MOTOR_IDS) -> None:
        """
        Disable motors and prevent future sends until clear_estop() is called.
        """
        if not self.estop:
            self.estop = True
            self.estop_reason = str(reason)
            self.estop_stamp = time.time()

        # best effort disable (don’t block on recv)
        for mid in motor_ids:
            try:
                self._send_cmd_ff(mid, CAN_CMD_DISABLE)
                print(f"[ESTOP] Disabled motor {mid} due to: {reason}")
            except Exception:
                pass

    # -------------------------
    # Low-level send
    # -------------------------
    def _send8(self, motor_id: int, data8: Iterable[int]) -> None:
        msg = can.Message(arbitration_id=int(motor_id), data=list(data8), is_extended_id=False)
        self.bus.send(msg)

    def _send_cmd_ff(self, motor_id: int, cmd: int) -> None:
        data = [0xFF] * 8
        data[7] = int(cmd)
        self._send8(motor_id, data)

    # -------------------------
    # State update
    # -------------------------
    def _try_update_state_from_msg(self, msg: can.Message) -> Optional[int]:
        try:
            raw = bytes(msg.data)
            mid = int(raw[0])
            spec = LEG_MOTORS.get(mid)
            if spec is None:
                return None

            motor_id, st = decode_state_frame(raw, pmax=spec.pmax, vmax=spec.vmax, tmax=spec.tmax)
            st.stamp = time.time()

            # update cache
            with self._state_lock:
                self.state[motor_id] = st

            # SAFETY: kill if state out of bounds
            if not self._check_state_in_bounds(motor_id, st.position_deg):
                self.emergency_stop(
                    reason=f"State out of bounds on motor {motor_id} ({LEG_MOTORS[motor_id].name}): "
                           f"{st.position_deg:.2f} deg not in {JOINT_LIMITS_DEG.get(motor_id)} (±{STATE_MARGIN_DEG} deg)",
                    motor_ids=LEG_MOTOR_IDS,
                )
            return motor_id
        except Exception:
            return None

    def request_state(
        self,
        motor_ids: Iterable[int] = LEG_MOTOR_IDS,
        *,
        settle_s: float = 0.0025,
        drain_timeout_s: float = 0.001,
        max_msgs: int = 64,
    ) -> List[int]:
        motor_ids = list(motor_ids)

        # Even in estop, allow request_state() (useful to diagnose)
        touched = set()

        for mid in motor_ids:
            data = [0xFF] * 7 + [CAN_CMD_CLEAR_FAULT]
            self._send8(mid, data)

        time.sleep(settle_s)

        recvd = 0
        start = time.time()
        while recvd < max_msgs and (time.time() - start) < 0.02 and len(touched) < len(motor_ids):
            msg = self.bus.recv(drain_timeout_s)
            if msg is None:
                break
            recvd += 1
            rec_mid = self._try_update_state_from_msg(msg)
            if rec_mid is not None:
                touched.add(rec_mid)

        missing = [mid for mid in motor_ids if mid not in touched]
        return missing

    # -------------------------
    # Basic commands
    # -------------------------
    def enable(self, motor_id: int) -> bool:
        self._assert_can_send()
        self._send_cmd_ff(motor_id, CAN_CMD_ENABLE)
        msg = self.bus.recv(self.recv_timeout_s)
        if msg is not None:
            self._try_update_state_from_msg(msg)
        return msg is not None

    def disable(self, motor_id: int) -> bool:
        self._send_cmd_ff(motor_id, CAN_CMD_DISABLE)
        msg = self.bus.recv(self.recv_timeout_s)
        if msg is not None:
            self._try_update_state_from_msg(msg)
        return msg is not None

    def set_zero(self, motor_id: int, *, wait_s: float = 0.5) -> Optional[can.Message]:
        self._assert_can_send()
        data = [0xFF] * 7 + [CAN_CMD_ZERO]
        self._send8(motor_id, data)
        msg = self.bus.recv(wait_s)
        if msg is not None:
            self._try_update_state_from_msg(msg)
        return msg

    # -------------------------
    # Gains
    # -------------------------
    def set_joint_gains(self, motor_id: int, *, kp: Optional[float] = None, kd: Optional[float] = None) -> None:
        g = self.gains[motor_id]
        self.gains[motor_id] = JointGains(kp if kp is not None else g.kp,
                                          kd if kd is not None else g.kd)

    # -------------------------
    # MIT control (with command safety)
    # -------------------------
    def mit_command(
        self,
        motor_id: int,
        *,
        position_deg: float,
        velocity_deg_s: float = 0.0,
        torque_nm: float = 0.0,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        recv: bool = True,
        clamp: bool = True,
    ) -> Optional[can.Message]:
        self._assert_can_send()

        # SAFETY: clamp commands to joint limits
        cmd_pos = self._clamp_command(motor_id, position_deg) if clamp else float(position_deg)

        spec = LEG_MOTORS[motor_id]
        g = self.gains[motor_id]
        kp_use = g.kp if kp is None else kp
        kd_use = g.kd if kd is None else kd

        payload = pack_mit_command(
            position_deg=cmd_pos,
            velocity_deg_s=velocity_deg_s,
            kp=kp_use,
            kd=kd_use,
            torque_nm=torque_nm,
            pmax=spec.pmax,
            vmax=spec.vmax,
            tmax=spec.tmax,
        )
        tx = can.Message(arbitration_id=int(motor_id), data=payload, is_extended_id=False)
        self.bus.send(tx)

        if not recv:
            return None

        rx = self.bus.recv(self.recv_timeout_s)
        if rx is not None:
            self._try_update_state_from_msg(rx)
        return rx

    def move_joint_deg(self, motor_id: int, pos_deg: float, *, recv: bool = False) -> None:
        self.mit_command(motor_id, position_deg=pos_deg, recv=recv)

    def move_leg_pose_deg(
        self,
        *,
        hipz: Optional[float] = None,
        hipx: Optional[float] = None,
        knee: Optional[float] = None,
        hipy: Optional[float] = None,
        ankle1: Optional[float] = None,
        ankle2: Optional[float] = None,
        recv: bool = False,
    ) -> None:
        mapping = {1: hipz, 2: hipx, 3: knee, 4: hipy, 5: ankle1, 6: ankle2}
        for mid, val in mapping.items():
            if val is None:
                continue
            self.move_joint_deg(mid, float(val), recv=recv)

    # -------------------------
    # Viz (unchanged except: respects estop by still displaying)
    # -------------------------
    def attach_meshcat(self, pin_viz) -> None:
        self._viz = pin_viz

    def _state_to_q(self) -> np.ndarray:
        q = np.zeros(6)
        with self._state_lock:
            s = dict(self.state)

        q[0] = np.deg2rad(s[1].position_deg)
        q[1] = np.deg2rad(s[2].position_deg)
        q[2] = np.deg2rad(s[4].position_deg)
        q[3] = np.deg2rad(s[3].position_deg)

        a1 = s[5].position_deg
        a2 = s[6].position_deg
        q[4] = -np.deg2rad((a1 - a2) / 2.0)
        q[5] =  np.deg2rad((a1 + a2) / 2.0)
        return q

    def start_viz(self, *, hz: float = 30.0, auto_request_state: bool = True) -> None:
        if self._viz is None:
            raise RuntimeError("No viz attached. Call robot.attach_meshcat(viz) first.")
        if self._viz_thread is not None:
            return

        self._viz_hz = float(hz)
        self._viz_stop = False
        self.auto_request_state = auto_request_state
        def loop():
            period = 1.0 / self._viz_hz
            while not self._viz_stop:
                if self.auto_request_state:
                    self.request_state(LEG_MOTOR_IDS)
                q = self._state_to_q()
                try:
                    self._viz.display(q)
                except Exception:
                    pass
                time.sleep(period)

        self._viz_thread = threading.Thread(target=loop, daemon=True)
        self._viz_thread.start()

    def stop_viz(self) -> None:
        self._viz_stop = True
        self._viz_thread = None
