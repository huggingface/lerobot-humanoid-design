"""
Motor utility functions for IPython sessions and scripts.

Typical IPython usage
---------------------
    import sys; sys.path.insert(0, "..")  # if running from tools/
    from tools.motor_utils import scan_motors, set_motor_zero, ping_motors_fast

    found = scan_motors()
    print(found)  # {'can0': [1,2,3,4,5,6], 'can1': [7,8,9,10,11,12]}
"""
from __future__ import annotations

from typing import Dict, List

import can

from .robstride_toolkit import ping_mit  # type: ignore

CAN_CMD_SET_ZERO = 0xFE
CAN_CMD_CLEAR_FAULT = 0xFB


def set_motor_zero(bus: can.BusABC, motor_id: int, timeout_s: float = 0.5) -> None:
    """
    Set the current encoder position as zero for *motor_id*.
    Motor must be enabled and stationary.
    """
    data = [0xFF] * 7 + [CAN_CMD_SET_ZERO]
    msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    bus.send(msg)
    reply = bus.recv(timeout_s)
    if reply is not None:
        print(f"m{motor_id}: zero set, reply={bytes(reply.data).hex()}")
    else:
        print(f"m{motor_id}: zero command sent (no reply within {timeout_s}s)")


def scan_motors(
    *,
    interface: str = "socketcan",
    channel_can0: str = "can0",
    channel_can1: str = "can1",
    motor_ids: range = range(1, 13),
) -> Dict[str, List[int]]:
    """
    Open both CAN buses, ping every motor ID, return which motors responded.

    Returns
    -------
    dict with keys 'can0' and 'can1', values are lists of responding motor IDs.
    """
    bus0 = can.interface.Bus(interface=interface, channel=channel_can0)
    bus1 = can.interface.Bus(interface=interface, channel=channel_can1)
    found: Dict[str, List[int]] = {channel_can0: [], channel_can1: []}
    try:
        for mid in motor_ids:
            if ping_mit(bus0, int(mid)):
                found[channel_can0].append(int(mid))
        for mid in motor_ids:
            if ping_mit(bus1, int(mid)):
                found[channel_can1].append(int(mid))
        return found
    finally:
        try:
            bus0.shutdown()
        except Exception:
            pass
        try:
            bus1.shutdown()
        except Exception:
            pass


def drain_bus(bus: can.BusABC, timeout_s: float = 0.1) -> int:
    """Drain all pending messages from *bus*. Returns number of messages drained."""
    count = 0
    while bus.recv(timeout_s) is not None:
        count += 1
    return count
