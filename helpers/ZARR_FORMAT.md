# PerAct2 Zarr Data Format

This document describes the zarr data format used for bimanual robot manipulation data.

## Overview

- **Bimanual setup**: 2 robot arms (left & right)
- **3 cameras**: front, wrist_left, wrist_right
- **Keyframe-based**: Data stored at important moments (gripper events, significant pose changes)

## Array Specifications

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `rgb` | (N, 3, 3, 256, 256) | uint8 | RGB images [cameras, channels, H, W] |
| `depth` | (N, 3, 256, 256) | float16 | Depth in meters [cameras, H, W] |
| `extrinsics` | (N, 3, 4, 4) | float16 | Camera-to-world transforms |
| `intrinsics` | (N, 3, 3, 3) | float16 | Camera matrices |
| `proprioception` | (N, 3, 2, 8) | float32 | EEF poses [history=3, hands=2, pose=8] |
| `action` | (N, 1, 2, 8) | float32 | Target EEF poses [1, hands=2, pose=8] |
| `proprioception_joints` | (N, 1, 2, 8) | float32 | Joint angles [1, hands=2, 8] |
| `action_joints` | (N, 1, 2, 8) | float32 | Target joint angles |
| `task_id` | (N,) | uint8 | Index into task list |
| `variation` | (N,) | uint8 | Task variation index |

## Data Formats

### Pose (8D)
```
[x, y, z, qx, qy, qz, qw, gripper_open]
```
- `x, y, z`: End-effector position in world frame
- `qx, qy, qz, qw`: Orientation quaternion (xyzw format)
- `gripper_open`: 1.0 = open, 0.0 = closed

### Joints (8D)
```
[j1, j2, j3, j4, j5, j6, j7, gripper_open]
```
- `j1-j7`: Joint angles in radians
- `gripper_open`: 1.0 = open, 0.0 = closed

### Camera Indices
```
0: front
1: wrist_left
2: wrist_right
```

### Hand Indices
```
0: left_arm
1: right_arm
```

### Intrinsics Matrix
```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```
Note: fx/fy may be negative depending on convention.

## Detailed Array Descriptions

### rgb
- Shape: `(N, 3, 3, 256, 256)`
- First dim 3 = cameras: [front, wrist_left, wrist_right]
- Second dim 3 = RGB channels
- 256x256 = image resolution

### depth
- Shape: `(N, 3, 256, 256)`
- Values in **meters**
- Same camera order as rgb

### proprioception
- Shape: `(N, 3, 2, 8)`
- Dim 3 = history: [t-2, t-1, t] (current + 2 previous poses)
- Dim 2 = hands: [left_arm, right_arm]
- Dim 8 = pose format above

### action
- Shape: `(N, 1, 2, 8)`
- TARGET pose for the **next keyframe**
- Same pose format as proprioception

### proprioception_joints / action_joints
- Shape: `(N, 1, 2, 8)`
- No history (just current state)
- 7 joint angles + gripper_open

## Task List

| task_id | Task Name |
|---------|-----------|
| 0 | bimanual_push_box |
| 1 | bimanual_lift_ball |
| 2 | bimanual_dual_push_buttons |
| 3 | bimanual_pick_plate |
| 4 | bimanual_put_item_in_drawer |
| 5 | bimanual_put_bottle_in_fridge |
| 6 | bimanual_handover_item |
| 7 | bimanual_pick_laptop |
| 8 | bimanual_straighten_rope |
| 9 | bimanual_sweep_to_dustpan |
| 10 | bimanual_lift_tray |
| 11 | bimanual_handover_item_easy |
| 12 | bimanual_take_tray_out_of_oven |

## Keyframe Semantics

Data is stored at **keyframes**, not every timestep. Keyframes are identified at:
- Gripper open/close events
- Significant pose changes
- Task-specific waypoints

At each keyframe:
- `proprioception[t]` = current + 2 previous EEF poses
- `action[t]` = target EEF pose for **next** keyframe
- `rgb/depth` = observation at current keyframe

## Creating Your Own Zarr

```python
import zarr
import numpy as np
from numcodecs import Blosc

compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
z = zarr.open('my_data.zarr', mode='w')

def create(name, shape, dtype):
    z.create_dataset(name, shape=(0,)+shape, chunks=(1,)+shape,
                     compressor=compressor, dtype=dtype)

# Create all arrays
create("rgb", (3, 3, 256, 256), "uint8")
create("depth", (3, 256, 256), "float16")
create("extrinsics", (3, 4, 4), "float16")
create("intrinsics", (3, 3, 3), "float16")
create("proprioception", (3, 2, 8), "float32")
create("action", (1, 2, 8), "float32")
create("proprioception_joints", (1, 2, 8), "float32")
create("action_joints", (1, 2, 8), "float32")
create("task_id", (), "uint8")
create("variation", (), "uint8")

# Append data (one keyframe at a time)
z['rgb'].append(rgb_data[np.newaxis])              # (1, 3, 3, 256, 256)
z['depth'].append(depth_data[np.newaxis])          # (1, 3, 256, 256)
z['extrinsics'].append(ext_data[np.newaxis])       # (1, 3, 4, 4)
z['intrinsics'].append(int_data[np.newaxis])       # (1, 3, 3, 3)
z['proprioception'].append(prop_data[np.newaxis])  # (1, 3, 2, 8)
z['action'].append(action_data[np.newaxis])        # (1, 1, 2, 8)
z['proprioception_joints'].append(prop_j[np.newaxis])   # (1, 1, 2, 8)
z['action_joints'].append(action_j[np.newaxis])    # (1, 1, 2, 8)
z['task_id'].append(np.array([task_id]))           # (1,)
z['variation'].append(np.array([variation]))       # (1,)
```

## Test Data Structure

```
peract2_test/
└── {task_name}/
    └── variation{N}/
        └── episodes/
            └── episode{M}/
                └── low_dim_obs.pkl
```

The `low_dim_obs.pkl` contains an RLBench `Demo` object with:
- `demo.left` / `demo.right`: Arm observations
  - `gripper_pose`: (7,) - [x, y, z, qx, qy, qz, qw]
  - `gripper_open`: float
  - `joint_positions`: (7,)
- `demo.misc`: Camera data (extrinsics, intrinsics, near/far planes)
