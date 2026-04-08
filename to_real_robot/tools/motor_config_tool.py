"""
Interactive motor configuration tool.

Run as a script:
    python tools/motor_config_tool.py

What it does:
1. Auto-detects the motor protocol (CANopen / private / MIT) via ping on can0.
2. If needed, switches the motor to MIT protocol (may require manual reboot).
3. Lets you change the motor ID.

Note: this tool works on a single motor at a time (one CAN bus, one ID).
For whole-robot scanning use tools/motor_utils.py::scan_motors().
"""
from __future__ import annotations

import sys
from time import sleep

import can

from .robstride_toolkit import (  # type: ignore
    ping_canopen,
    ping_private,
    ping_mit,
    switch_canopen_to_private,
    switch_private_to_mit,
    change_motor_id,
)

CAN_CMD_SET_ZERO = 0xFE


def _detect_motor(bus: can.BusABC) -> tuple[int | None, str | None]:
    """Return (motor_id, protocol) or (None, None) if nothing responds."""
    for proto, ping_fn in [
        ("CANopen", ping_canopen),
        ("private", ping_private),
        ("MIT",     ping_mit),
    ]:
        for mid in range(128):
            if ping_fn(bus, mid):
                print(f"  Motor {mid} responded to {proto} ping.")
                return mid, proto
    return None, None


def run() -> None:
    bus = can.interface.Bus(interface="socketcan", channel="can0")

    print("Scanning for motor on can0 ...")
    motor_id, protocol = _detect_motor(bus)

    if motor_id is None:
        print("No motor responded on can0.")
        bus.shutdown()
        sys.exit(1)

    print(f"Detected motor_id={motor_id}, protocol={protocol}")

    # Switch to MIT protocol if not already there
    if protocol == "CANopen":
        print("Switching CANopen -> private ...")
        switch_canopen_to_private(bus, motor_id)
        sleep(1.0)
        input("Reboot motor, then press Enter ...")
        print("Switching private -> MIT ...")
        switch_private_to_mit(bus, motor_id)
        input("Reboot motor again, then press Enter ...")
    elif protocol == "private":
        print("Switching private -> MIT ...")
        switch_private_to_mit(bus, motor_id)
        input("Reboot motor, then press Enter ...")

    # Change motor ID
    print(f"Current motor ID: {motor_id}")
    new_id_str = input("Enter new motor ID (1-127), or press Enter to skip: ").strip()
    if new_id_str:
        new_id = int(new_id_str)
        if 1 <= new_id <= 127:
            print(f"Changing motor ID {motor_id} -> {new_id} ...")
            change_motor_id(bus, motor_id, new_id)
            motor_id = new_id
        else:
            print("Invalid ID, skipping.")

    # Optionally set encoder zero
    zero_str = input("Set current position as encoder zero? [y/N]: ").strip().lower()
    if zero_str == "y":
        data = [0xFF] * 7 + [CAN_CMD_SET_ZERO]
        msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        bus.send(msg)
        reply = bus.recv(0.5)
        print(f"Zero set. Reply: {bytes(reply.data).hex() if reply else 'none'}")

    bus.shutdown()
    print("Done.")


if __name__ == "__main__":
    run()
