from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


# AGILE (Isaac Lab) observation term names → RL_agent_isolated canonical names.
_AGILE_TERM_RENAME = {
    "velocity_commands": "command",
    "controlled_joint_pos": "joint_pos",
    "controlled_joint_vel": "joint_vel",
}
# Keys on observations.policy that sit alongside the terms (i.e. not themselves terms).
_POLICY_META_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}


def _sanitize_for_safe_yaml(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_safe_yaml(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_safe_yaml(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)) or value is None:
        return value
    # slice, type, function — unrepresentable; drop.
    return None


def adapt_agile_env_to_mjlab_config(env_yaml_path: Path, config_yaml_path: Path) -> None:
    """Translate an AGILE-exported env.yaml into the mjlab-convention config.yaml
    that to_real_robot/RL_agent_isolated.py expects.

    AGILE puts obs terms directly under observations.policy.<term_name> with names
    like velocity_commands / controlled_joint_{pos,vel}, and keeps robot config at
    scene.robot. RL_agent_isolated looks for observations.policy.terms.<term_name>
    with canonical names (command, joint_pos, joint_vel) and scene.entities.robot.
    """
    import yaml  # type: ignore

    with open(env_yaml_path) as f:
        env = yaml.unsafe_load(f)
    if not isinstance(env, dict):
        raise ValueError(f"env.yaml did not parse into a dict: {env_yaml_path}")

    obs = env.get("observations")
    if isinstance(obs, dict):
        for group_name in ("policy", "critic"):
            group = obs.get(group_name)
            if not isinstance(group, dict):
                continue
            terms: dict = {}
            meta: dict = {}
            for k, v in group.items():
                if k in _POLICY_META_KEYS:
                    meta[k] = v
                else:
                    terms[_AGILE_TERM_RENAME.get(k, k)] = v
            rebuilt = dict(meta)
            rebuilt["terms"] = terms
            obs[group_name] = rebuilt

    scene = env.get("scene")
    if isinstance(scene, dict) and "robot" in scene and "entities" not in scene:
        scene["entities"] = {"robot": scene.pop("robot")}

    policy_group = (
        env.get("observations", {}).get("policy")
        if isinstance(env.get("observations"), dict)
        else None
    )
    if isinstance(policy_group, dict) and "history_length" in policy_group:
        env.setdefault("history_len", policy_group["history_length"])

    # Mark the config so RL_agent_isolated uses the Isaac Lab observation layout:
    # terms stay in training insertion order (no canonical reordering) and the
    # history vector is term-major instead of time-major.
    env["_obs_layout"] = "isaaclab"

    clean = _sanitize_for_safe_yaml(env)
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)


def load_hf_policy(
    hf_repo: str,
    hf_subdir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Download a HF policy snapshot, locate its policy.onnx + env.yaml/config.yaml,
    translate env.yaml → mjlab-format config.yaml when needed, and return
    (policy_dir, config_path, policy_path) ready to pass to RLAgent.from_files.
    """
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(hf_repo))

    def _looks_like_policy_dir(d: Path) -> bool:
        return (d / "policy.onnx").is_file() and (
            (d / "config.yaml").is_file() or (d / "env.yaml").is_file()
        )

    if hf_subdir:
        policy_dir = snap / hf_subdir
    elif _looks_like_policy_dir(snap):
        policy_dir = snap
    else:
        candidates = [d for d in snap.iterdir() if d.is_dir() and _looks_like_policy_dir(d)]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one policy subdir in {snap}, got "
                f"{[d.name for d in candidates]}"
            )
        policy_dir = candidates[0]

    config_path = policy_dir / "config.yaml"
    if not config_path.is_file() and (policy_dir / "env.yaml").is_file():
        adapt_agile_env_to_mjlab_config(policy_dir / "env.yaml", config_path)

    return policy_dir, config_path, policy_dir / "policy.onnx"
