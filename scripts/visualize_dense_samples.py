"""
Visualize dense trajectory samples with RGB, depth, proprio, and action trajectories.

Usage:
    python scripts/visualize_dense_samples.py \
        --src bimanual_push_box/Peract2_zarr_dense/train.zarr \
        --samples 0 1 \
        --output /tmp/sample_viz
"""

import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Path to dense zarr')
    parser.add_argument('--samples', type=int, nargs='+', default=[0, 1], help='Sample indices to visualize')
    parser.add_argument('--output', type=str, default='/tmp/sample_viz', help='Output directory')
    return parser.parse_args()


def visualize_sample(z, sample_idx, output_dir):
    """Create comprehensive visualization for a single sample."""

    rgb = z['rgb'][sample_idx]           # (3, 3, 256, 256) - 3 cameras
    depth = z['depth'][sample_idx]       # (3, 256, 256)
    action = z['action'][sample_idx]     # (15, 2, 8)
    proprio = z['proprioception'][sample_idx]  # (3, 2, 8)

    traj_len = action.shape[0]

    fig = plt.figure(figsize=(20, 16))

    # ===== ROW 1: RGB images (3 cameras) + depth =====
    for cam in range(3):
        ax = fig.add_subplot(4, 4, cam + 1)
        img = rgb[cam].transpose(1, 2, 0)  # (3, 256, 256) -> (256, 256, 3)
        img = np.clip(img, 0, 1) if img.max() <= 1 else np.clip(img / 255, 0, 1)
        ax.imshow(img)
        ax.set_title(f'RGB Camera {cam}')
        ax.axis('off')

    ax = fig.add_subplot(4, 4, 4)
    ax.imshow(depth[0], cmap='viridis')
    ax.set_title('Depth Camera 0')
    ax.axis('off')

    # ===== ROW 2: Proprioception history =====
    # Left arm proprio history
    ax = fig.add_subplot(4, 4, 5)
    for t in range(3):
        pos = proprio[t, 0, :3]
        ax.scatter(t, pos[0], c='r', marker='o', s=100, label='X' if t == 0 else '')
        ax.scatter(t, pos[1], c='g', marker='s', s=100, label='Y' if t == 0 else '')
        ax.scatter(t, pos[2], c='b', marker='^', s=100, label='Z' if t == 0 else '')
    ax.set_xlabel('History frame')
    ax.set_ylabel('Position')
    ax.set_title('Proprio History (Left Arm)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['t-2', 't-1', 't (current)'])
    ax.legend()
    ax.grid(True)

    # Right arm proprio history
    ax = fig.add_subplot(4, 4, 6)
    for t in range(3):
        pos = proprio[t, 1, :3]
        ax.scatter(t, pos[0], c='r', marker='o', s=100, label='X' if t == 0 else '')
        ax.scatter(t, pos[1], c='g', marker='s', s=100, label='Y' if t == 0 else '')
        ax.scatter(t, pos[2], c='b', marker='^', s=100, label='Z' if t == 0 else '')
    ax.set_xlabel('History frame')
    ax.set_ylabel('Position')
    ax.set_title('Proprio History (Right Arm)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['t-2', 't-1', 't (current)'])
    ax.legend()
    ax.grid(True)

    # Current state text
    ax = fig.add_subplot(4, 4, 7)
    ax.axis('off')
    text = "Proprio (current, t=-1):\n"
    text += f"Left:  XYZ={proprio[-1, 0, :3].round(3)}\n"
    text += f"       Quat={proprio[-1, 0, 3:7].round(3)}\n"
    text += f"Right: XYZ={proprio[-1, 1, :3].round(3)}\n"
    text += f"       Quat={proprio[-1, 1, 3:7].round(3)}"
    ax.text(0.1, 0.5, text, fontsize=10, family='monospace', va='center')
    ax.set_title('Current State')

    # ===== ROW 3: Left arm trajectory =====
    steps = np.arange(traj_len)
    left = action[:, 0, :]
    right = action[:, 1, :]

    # Calculate deltas
    left_delta = left[-1, :3] - left[0, :3]
    right_delta = right[-1, :3] - right[0, :3]

    # Left arm XYZ
    ax = fig.add_subplot(4, 4, 9)
    ax.plot(steps, left[:, 0], 'r-o', label='X', markersize=4)
    ax.plot(steps, left[:, 1], 'g-s', label='Y', markersize=4)
    ax.plot(steps, left[:, 2], 'b-^', label='Z', markersize=4)
    ax.axhline(proprio[-1, 0, 0], c='r', ls='--', alpha=0.5)
    ax.axhline(proprio[-1, 0, 1], c='g', ls='--', alpha=0.5)
    ax.axhline(proprio[-1, 0, 2], c='b', ls='--', alpha=0.5)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Position (m)')
    title = f'Left Arm: XYZ Trajectory ({traj_len} steps)\n'
    title += f'ΔX={left_delta[0]:.3f}m  ΔY={left_delta[1]:.3f}m  ΔZ={left_delta[2]*1000:.1f}mm'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Left arm quaternion
    ax = fig.add_subplot(4, 4, 10)
    ax.plot(steps, left[:, 3], '-o', label='qx', markersize=4)
    ax.plot(steps, left[:, 4], '-s', label='qy', markersize=4)
    ax.plot(steps, left[:, 5], '-^', label='qz', markersize=4)
    ax.plot(steps, left[:, 6], '-d', label='qw', markersize=4)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Quaternion')
    ax.set_title('Left Arm: Rotation Trajectory')
    ax.legend()
    ax.grid(True)

    # ===== ROW 4: Right arm trajectory =====
    # Right arm XYZ
    ax = fig.add_subplot(4, 4, 13)
    ax.plot(steps, right[:, 0], 'r-o', label='X', markersize=4)
    ax.plot(steps, right[:, 1], 'g-s', label='Y', markersize=4)
    ax.plot(steps, right[:, 2], 'b-^', label='Z', markersize=4)
    ax.axhline(proprio[-1, 1, 0], c='r', ls='--', alpha=0.5)
    ax.axhline(proprio[-1, 1, 1], c='g', ls='--', alpha=0.5)
    ax.axhline(proprio[-1, 1, 2], c='b', ls='--', alpha=0.5)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Position (m)')
    title = f'Right Arm: XYZ Trajectory ({traj_len} steps)\n'
    title += f'ΔX={right_delta[0]:.3f}m  ΔY={right_delta[1]:.3f}m  ΔZ={right_delta[2]*1000:.1f}mm'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Right arm quaternion
    ax = fig.add_subplot(4, 4, 14)
    ax.plot(steps, right[:, 3], '-o', label='qx', markersize=4)
    ax.plot(steps, right[:, 4], '-s', label='qy', markersize=4)
    ax.plot(steps, right[:, 5], '-^', label='qz', markersize=4)
    ax.plot(steps, right[:, 6], '-d', label='qw', markersize=4)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Quaternion')
    ax.set_title('Right Arm: Rotation Trajectory')
    ax.legend()
    ax.grid(True)

    # 3D trajectory plot
    ax = fig.add_subplot(4, 4, 15, projection='3d')
    ax.plot(left[:, 0], left[:, 1], left[:, 2], 'r-o', label='Left', markersize=4)
    ax.plot(right[:, 0], right[:, 1], right[:, 2], 'b-o', label='Right', markersize=4)
    ax.scatter(*proprio[-1, 0, :3], c='r', s=150, marker='*', label='Left start')
    ax.scatter(*proprio[-1, 1, :3], c='b', s=150, marker='*', label='Right start')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory')
    ax.legend()

    # Gripper state
    ax = fig.add_subplot(4, 4, 16)
    ax.plot(steps, left[:, 7], 'r-o', label='Left gripper', markersize=4)
    ax.plot(steps, right[:, 7], 'b-s', label='Right gripper', markersize=4)
    ax.set_xlabel('Trajectory step')
    ax.set_ylabel('Gripper (0=closed, 1=open)')
    ax.set_title('Gripper State')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    ax.grid(True)

    plt.suptitle(f'Sample {sample_idx} - Full Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'sample_{sample_idx}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.src}")
    z = zarr.open(args.src, 'r')

    print(f"Data shapes:")
    for k in z.keys():
        print(f"  {k}: {z[k].shape}")

    print(f"\nVisualizing samples: {args.samples}")
    for sample_idx in args.samples:
        if sample_idx >= len(z['action']):
            print(f"Warning: Sample {sample_idx} out of range, skipping")
            continue
        visualize_sample(z, sample_idx, args.output)

    print(f"\nDone! Output in: {args.output}")


if __name__ == '__main__':
    main()
