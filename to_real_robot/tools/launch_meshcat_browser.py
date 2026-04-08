#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Dict, Optional


def _resolve_meshcat_url(explicit_url: Optional[str], zmq_url: str) -> str:
    if explicit_url:
        return explicit_url

    try:
        import meshcat  # type: ignore

        vis = meshcat.Visualizer(zmq_url=zmq_url)
        url_attr = getattr(vis, "url", None)
        if callable(url_attr):
            return str(url_attr())
        if isinstance(url_attr, str):
            return url_attr
    except Exception:
        pass

    return "http://127.0.0.1:7000/static/"


def _wait_for_http(url: str, timeout_s: float, poll_s: float) -> bool:
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            if time.time() >= deadline:
                return False
            time.sleep(max(0.05, poll_s))


def _pick_browser(requested: str) -> str:
    if requested != "auto":
        path = shutil.which(requested)
        if path:
            return requested
        raise RuntimeError(
            f"Requested browser '{requested}' was not found in PATH. "
            "Use --browser auto or an installed command (chromium/firefox/xdg-open)."
        )

    for candidate in ("chromium-browser", "chromium", "firefox", "xdg-open"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("No browser found. Install chromium-browser/chromium/firefox.")


def _build_desktop_env(display: str, xauthority: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    env["DISPLAY"] = display

    if xauthority:
        env["XAUTHORITY"] = xauthority
    else:
        default_xauth = Path.home() / ".Xauthority"
        if default_xauth.exists():
            env.setdefault("XAUTHORITY", str(default_xauth))

    uid = os.getuid()
    runtime_dir = f"/run/user/{uid}"
    if os.path.isdir(runtime_dir):
        env.setdefault("XDG_RUNTIME_DIR", runtime_dir)
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
        env.setdefault("WAYLAND_DISPLAY", "wayland-0")

    return env


def _launch(browser: str, url: str, kiosk: bool, zoom: float, env: Dict[str, str]) -> subprocess.Popen:
    if browser in ("chromium-browser", "chromium"):
        cmd = [
            browser,
            "--new-window",
            "--disable-session-crashed-bubble",
            "--no-first-run",
            "--noerrdialogs",
            "--no-default-browser-check",
            "--disable-save-password-bubble",
            "--password-store=basic",
            "--use-mock-keychain",
            "--disable-features=PasswordManagerOnboarding,PasswordManagerRedesign,AutofillServerCommunication",
        ]
        if abs(float(zoom) - 1.0) > 1e-6:
            cmd.append(f"--force-device-scale-factor={float(zoom):.3f}")
        if kiosk:
            cmd.append("--kiosk")
        cmd.append(url)
    elif browser == "firefox":
        cmd = [browser]
        if kiosk:
            cmd.append("--kiosk")
        cmd.append(url)
    else:
        cmd = [browser, url]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _window_class_for_browser(browser: str) -> Optional[str]:
    if browser in ("chromium-browser", "chromium"):
        return "chromium"
    if browser == "firefox":
        return "firefox"
    return None


def _close_existing_chromium(env: Dict[str, str]) -> None:
    # Best-effort cleanup so kiosk startup is deterministic.
    for name in ("chromium", "chromium-browser"):
        subprocess.run(
            ["pkill", "-x", name],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _find_window_id(window_class: str, env: Dict[str, str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["xdotool", "search", "--onlyvisible", "--class", window_class],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _get_window_center(window_id: str, env: Dict[str, str]) -> tuple[int, int]:
    proc = subprocess.run(
        ["xdotool", "getwindowgeometry", "--shell", window_id],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return (640, 360)
    width = 1280
    height = 720
    for line in proc.stdout.splitlines():
        if line.startswith("WIDTH="):
            try:
                width = int(line.split("=", 1)[1])
            except Exception:
                pass
        elif line.startswith("HEIGHT="):
            try:
                height = int(line.split("=", 1)[1])
            except Exception:
                pass
    return (max(1, width // 2), max(1, height // 2))


def _apply_wheel_zoom(browser: str, wheel_steps: int, wheel_delay_s: float, env: Dict[str, str]) -> None:
    if wheel_steps == 0:
        return
    if shutil.which("xdotool") is None:
        print("[ERROR] wheel zoom requested but xdotool is not installed.")
        return

    session_type = env.get("XDG_SESSION_TYPE", os.environ.get("XDG_SESSION_TYPE", "")).lower()
    if session_type == "wayland":
        print("[WARN] xdotool may not work on Wayland sessions.")

    time.sleep(max(0.0, wheel_delay_s))
    while time.time() < time.time() + 30.0 :  # retry for a few seconds if window not found
        window_class = _window_class_for_browser(browser)
        window_id = _find_window_id(window_class, env) if window_class else None
        if window_id:
            break
        time.sleep(0.5)
    if not window_id:
        print("[WARN] Could not find browser window for automated zoom.")
        return

    subprocess.run(
        ["xdotool", "windowactivate", "--sync", window_id],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    cx, cy = _get_window_center(window_id, env)
    subprocess.run(
        ["xdotool", "mousemove", "--window", window_id, str(cx), str(cy)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    wheel_button = "4" if wheel_steps > 0 else "5"
    for _ in range(abs(int(wheel_steps))):
        subprocess.run(
            ["xdotool", "click", "--window", window_id, wheel_button],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch MeshCat webpage on Raspberry Pi display."
    )
    parser.add_argument("--url", type=str, default=None, help="Explicit webpage URL to open.")
    parser.add_argument(
        "--zmq-url",
        type=str,
        default="tcp://127.0.0.1:6000",
        help="MeshCat ZMQ URL used to infer webpage URL when --url is not provided.",
    )
    parser.add_argument(
        "--browser",
        type=str,
        default="auto",
        help="Browser command (auto/chromium-browser/chromium/firefox/xdg-open).",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for URL to become reachable (0 to skip waiting).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.2,
        help="Polling interval in seconds while waiting for URL.",
    )
    parser.add_argument(
        "--no-kiosk",
        action="store_true",
        help="Open browser windowed instead of kiosk fullscreen.",
    )
    parser.add_argument(
        "--display",
        type=str,
        default=":0",
        help="Desktop display to target when launched from SSH.",
    )
    parser.add_argument(
        "--xauthority",
        type=str,
        default=None,
        help="Path to Xauthority file (for X11 desktops).",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Initial page zoom for Chromium via device scale factor (1.0 = default).",
    )
    parser.add_argument(
        "--wheel-steps",
        type=int,
        default=0,
        help="Simulate Ctrl+mouse-wheel steps after launch (positive=zoom in, negative=zoom out).",
    )
    parser.add_argument(
        "--wheel-delay",
        type=float,
        default=1.2,
        help="Seconds to wait after launch before sending wheel events.",
    )
    args = parser.parse_args()

    target_url = _resolve_meshcat_url(args.url, args.zmq_url)
    try:
        browser = _pick_browser(args.browser)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 2
    launch_env = _build_desktop_env(args.display, args.xauthority)

    if args.wait_timeout > 0.0:
        ok = _wait_for_http(target_url, timeout_s=args.wait_timeout, poll_s=args.poll_interval)
        if not ok:
            print(f"[WARN] URL not reachable after {args.wait_timeout:.1f}s: {target_url}")
            print("[INFO] Launching browser anyway.")

    if args.zoom <= 0.0:
        print("[ERROR] --zoom must be > 0.")
        return 2
    if args.wheel_steps != 0 and shutil.which("xdotool") is None:
        print("[ERROR] --wheel-steps requires xdotool. Install it: sudo apt-get install -y xdotool")
        return 2

    _close_existing_chromium(launch_env)
    proc = _launch(browser, target_url, kiosk=not args.no_kiosk, zoom=args.zoom, env=launch_env)
    time.sleep(0.8)
    rc = proc.poll()
    if rc is not None and rc != 0:
        err = (proc.stderr.read() or "").strip() if proc.stderr is not None else ""
        print(f"[ERROR] Browser exited immediately (code={rc}).")
        if err:
            print(err)
        return rc
    _apply_wheel_zoom(browser, args.wheel_steps, args.wheel_delay, launch_env)

    print(
        f"[OK] Launched {browser} on DISPLAY={launch_env.get('DISPLAY', '?')} "
        f"(zoom={args.zoom:.2f}, wheel_steps={args.wheel_steps}) -> {target_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
