"""
Replay a bipedal_state_log.csv in the MuJoCo sim with fixed base.

The log's m*_target_pos_deg columns (raw motor encoder space) are written
directly into sim_robot.action[mid].position_deg, which is exactly what
the sim's position-actuator control loop expects.

Usage (IPython):
    from replay_log import replay
    replay()                                     # defaults
    replay("debug_logs/bipedal_state_log.csv")  # explicit path
    replay(..., realtime=False)                  # as fast as sim allows
    replay(..., fixed_base=False)                # let robot fall freely

Usage (CLI):
    python replay_log.py
    python replay_log.py debug_logs/bipedal_state_log.csv
    python replay_log.py --no-realtime --no-fixed-base
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

from sim_robot import MotorCommand, SimBipedalRobotController
from root_constant import MOTOR_IDS

DEFAULT_LOG = Path("debug_logs/bipedal_state_log.csv")

# Column names in the log for raw motor target positions.
_TARGET_COL = {mid: f"m{mid}_target_pos_deg" for mid in MOTOR_IDS}


def load_log(path: Path) -> List[Dict]:
    """Load all rows from the log CSV. Returns list of dicts with keys:
    - time_s: float
    - mode: str
    - targets: {motor_id: float}  (raw encoder degrees, as commanded)
    """
    rows: List[Dict] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                targets = {mid: float(raw[col]) for mid, col in _TARGET_COL.items()}
            except (KeyError, ValueError):
                continue
            rows.append({
                "time_s": float(raw.get("time_s", 0.0)),
                "mode": str(raw.get("mode", "control")).strip(),
                "targets": targets,
            })
    return rows


def replay(
    csv_path: Path | str = DEFAULT_LOG,
    *,
    realtime: bool = True,
    fixed_base: bool = True,
    fixed_base_height_m: float = 0.77,
    control_hz: float = 200.0,
    skip_state_only: bool = True,
) -> None:
    """
    Replay a bipedal_state_log.csv in the MuJoCo sim.

    Args:
        csv_path:            Path to the log file.
        realtime:            If True, sleep between frames to match original timing.
                             If False, replay as fast as the sim allows.
        fixed_base:          If True, pin the pelvis in place (good for inspection).
        fixed_base_height_m: Height of the pinned pelvis (default 0.77 m).
        control_hz:          Sim control loop rate.
        skip_state_only:     If True, skip rows logged before control was enabled
                             (rows where mode != "control").
    """
    csv_path = Path(csv_path)
    print(f"[replay] loading {csv_path} ...")
    rows = load_log(csv_path)
    if not rows:
        print("[replay] no rows found in log.")
        return

    if skip_state_only:
        control_rows = [r for r in rows if r["mode"] == "control"]
        if control_rows:
            rows = control_rows
        else:
            print("[replay] no 'control' mode rows found; replaying all rows.")

    # Further filter rows where all targets are exactly 0 (pre-command state_only).
    non_zero = [r for r in rows if any(v != 0.0 for v in r["targets"].values())]
    if non_zero:
        rows = non_zero
    else:
        print("[replay] warning: all target positions are 0 — replaying anyway.")

    print(f"[replay] {len(rows)} frames | realtime={realtime} | fixed_base={fixed_base}")

    # Build sim.
    robot = SimBipedalRobotController(
        control_hz=control_hz,
        fixed_base=fixed_base,
        fixed_base_height_m=fixed_base_height_m,
        auto_reset_on_flip=False,
        auto_reset_on_divergence=False,
    )
    robot.start(mode="control", auto_enable=True)
    robot.start_viewer()

    # Pre-load first frame so the sim settles before replay begins.
    _write_targets(robot, rows[0]["targets"])
    time.sleep(0.5)

    log_t0 = rows[0]["time_s"]
    wall_t0 = time.perf_counter()

    for i, row in enumerate(rows):
        _write_targets(robot, row["targets"])

        if realtime and i + 1 < len(rows):
            # Sleep until the wall-clock moment that matches the next log timestamp.
            dt_log = rows[i + 1]["time_s"] - row["time_s"]
            target = wall_t0 + (row["time_s"] - log_t0) + dt_log
            sleep_s = target - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    print("[replay] done.")


def _write_targets(robot: SimBipedalRobotController, targets: Dict[int, float]) -> None:
    """Write raw motor target positions directly into the sim action dict."""
    with robot._action_lock:
        for mid in MOTOR_IDS:
            prev = robot.action[mid]
            robot.action[mid] = MotorCommand(
                position_deg=targets[mid],
                velocity_deg_s=0.0,
                torque_nm=0.0,
                kp=prev.kp,
                kd=prev.kd,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", nargs="?", default=str(DEFAULT_LOG), help="Path to bipedal_state_log.csv")
    p.add_argument("--no-realtime", dest="realtime", action="store_false", help="Replay as fast as sim allows")
    p.add_argument("--no-fixed-base", dest="fixed_base", action="store_false", help="Let robot fall freely")
    p.add_argument("--height", type=float, default=0.77, metavar="M", help="Fixed base height in metres (default 0.77)")
    p.add_argument("--hz", type=float, default=200.0, help="Sim control rate (default 200)")
    args = p.parse_args()
    replay(
        args.csv_path,
        realtime=args.realtime,
        fixed_base=args.fixed_base,
        fixed_base_height_m=args.height,
        control_hz=args.hz,
    )
