"""
Create dense trajectory zarr from keypose zarr for a single task.
Assumes all data is from one continuous task (no episode detection needed).

Usage:
    python data_processing/create_dense_trajectories.py \
        --src bimanual_push_box/Peract2_zarr/train.zarr \
        --tgt bimanual_push_box/Peract2_zarr_dense/train.zarr \
        --traj_len 15
"""

import argparse
import os
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from data_processing.rlbench_utils import (
    interpolate_trajectory,
    quat_to_euler_np,
    euler_to_quat_np
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source zarr with keyposes')
    parser.add_argument('--tgt', type=str, required=True, help='Target zarr for dense trajectories')
    parser.add_argument('--traj_len', type=int, default=15, help='Trajectory length (interpolation steps)')
    return parser.parse_args()


def interpolate_bimanual(current_pose, next_pose, num_steps):
    """
    Interpolate trajectory between two bimanual poses.

    Args:
        current_pose: (2, 8) - [left, right] each with [x,y,z, qx,qy,qz,qw, gripper]
        next_pose: (2, 8)

    Returns:
        (num_steps, 2, 8) - interpolated trajectory
    """
    def _interp_single_arm(start, end):
        # Stack start and end: (2, 8)
        traj = np.stack([start, end], axis=0)

        # Convert quat to euler for smooth interpolation
        traj_euler = np.concatenate([
            traj[:, :3],                    # xyz
            quat_to_euler_np(traj[:, 3:7]), # quat -> euler
            traj[:, 7:]                     # gripper
        ], axis=1)  # (2, 7)

        # Interpolate
        traj_interp = interpolate_trajectory(traj_euler, num_steps)  # (num_steps, 7)

        # Convert euler back to quat
        traj_quat = np.concatenate([
            traj_interp[:, :3],                    # xyz
            euler_to_quat_np(traj_interp[:, 3:6]), # euler -> quat
            traj_interp[:, 6:]                     # gripper
        ], axis=1)  # (num_steps, 8)

        return traj_quat

    left_traj = _interp_single_arm(current_pose[0], next_pose[0])
    right_traj = _interp_single_arm(current_pose[1], next_pose[1])

    return np.stack([left_traj, right_traj], axis=1)  # (num_steps, 2, 8)


def main():
    args = parse_arguments()

    if os.path.exists(args.tgt):
        print(f"Target {args.tgt} already exists. Delete it first or choose different path.")
        return

    # Load source
    print(f"Loading: {args.src}")
    src = zarr.open(args.src, 'r')

    actions = src['action'][:]  # (N, 1, 2, 8)
    N = len(actions)

    print(f"Source: {N} keyposes")
    print(f"Output: {N-1} samples with trajectory length {args.traj_len}")

    # Create target
    os.makedirs(os.path.dirname(args.tgt) or '.', exist_ok=True)
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)

    with zarr.open_group(args.tgt, mode='w') as tgt:
        # Helper to create dataset
        def _create(name, shape, dtype):
            tgt.create_dataset(
                name, shape=(0,) + shape, chunks=(1,) + shape,
                compressor=compressor, dtype=dtype
            )

        # Create datasets with appropriate shapes
        # Observations stay the same shape (single frame)
        _create('rgb', src['rgb'].shape[1:], 'uint8')
        _create('depth', src['depth'].shape[1:], 'float16')
        _create('proprioception', src['proprioception'].shape[1:], 'float32')

        # Action gets trajectory dimension
        _create('action', (args.traj_len, 2, 8), 'float32')

        # Metadata stays scalar
        _create('task_id', (), 'int32')
        _create('variation', (), 'int32')

        # Copy other fields if present (keep original shapes)
        for field in ['intrinsics', 'extrinsics', 'action_joints', 'proprioception_joints']:
            if field in src:
                _create(field, src[field].shape[1:], src[field].dtype)

        # Process each transition (keypose i -> keypose i+1)
        print("\nInterpolating trajectories...")
        for i in tqdm(range(N - 1)):
            current_pose = actions[i, 0]      # (2, 8)
            next_pose = actions[i + 1, 0]     # (2, 8)

            # Interpolate trajectory
            traj = interpolate_bimanual(current_pose, next_pose, args.traj_len)

            # Append to target
            # Observation at keypose i, action is trajectory to keypose i+1
            tgt['rgb'].append(src['rgb'][i:i+1])
            tgt['depth'].append(src['depth'][i:i+1])
            tgt['proprioception'].append(src['proprioception'][i:i+1])
            tgt['action'].append(traj[np.newaxis])  # (1, traj_len, 2, 8)
            tgt['task_id'].append(src['task_id'][i:i+1])
            tgt['variation'].append(src['variation'][i:i+1])

            for field in ['intrinsics', 'extrinsics', 'action_joints', 'proprioception_joints']:
                if field in src:
                    tgt[field].append(src[field][i:i+1])

    # Verify
    print(f"\nCreated: {args.tgt}")
    with zarr.open(args.tgt, 'r') as check:
        print("Shapes:")
        for k in sorted(check.keys()):
            print(f"  {k}: {check[k].shape}")


if __name__ == '__main__':
    main()
