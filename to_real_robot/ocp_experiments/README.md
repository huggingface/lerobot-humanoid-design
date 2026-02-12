# OCP Experiments (Real Robot)

This folder is the workspace for optimal-control experiments on:
`to_real_robot/model/urdf/robot.urdf`

## Jump OCP

Files:
- `jump/jump_loader.py`: loads the robot model and Sobec wrapper
- `jump/jump_params.py`: jump phase timing and cost weights
- `jump/jump_ocp.py`: builds and solves the jump OCP
- `jump/jump_fconf.yaml`: sample config file

Run from repo root:

```bash
python lerobot-humanoid-design/to_real_robot/ocp_experiments/jump/jump_ocp.py \
  --config lerobot-humanoid-design/to_real_robot/ocp_experiments/jump/jump_fconf.yaml \
  --save /tmp/real_robot_jump.npy
```

Optional visualization:

```bash
python lerobot-humanoid-design/to_real_robot/ocp_experiments/jump/jump_ocp.py --visualize
```
