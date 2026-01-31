"""Visualize saved trajectories using Matplotlib 3D plots."""

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory_3d(ax, traj, label, color, linestyle='-', linewidth=0.8, alpha=0.8,
                       show_waypoints=False, waypoint_size=3):
    """Plot a single trajectory on a 3D axis."""
    if len(traj) == 0:
        return

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            label=label, color=color, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)

    # Show all waypoints as small spheres
    if show_waypoints:
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2],
                   color=color, s=waypoint_size, alpha=0.6)

    # Start marker (larger, with border)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
               color=color, s=40, marker='o', edgecolors='black', linewidths=0.5)

    # End marker (diamond)
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
               color=color, s=60, marker='D', edgecolors='black', linewidths=0.5)


def setup_3d_axis(ax, title):
    """Configure 3D axis appearance."""
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.set_facecolor('white')
    ax.legend(loc='upper left', fontsize=7)


def visualize_episode(npz_path, show=True):
    """Visualize trajectories from a single episode with three separate comparison figures."""
    data = np.load(npz_path, allow_pickle=True)

    gt_left = data['gt_left']
    gt_right = data['gt_right']
    pred_left = data['pred_left']
    pred_right = data['pred_right']
    exec_left = data['exec_left']
    exec_right = data['exec_right']
    success = data['success']
    max_reward = data['max_reward']

    # Extract episode info from path
    episode_name = os.path.basename(npz_path).replace('_trajectories.npz', '')
    variation = os.path.basename(os.path.dirname(npz_path))
    task = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))

    success_str = "SUCCESS" if success else "FAILURE"

    # Colors
    gt_color = 'blue'
    pred_color = 'red'
    exec_color = 'green'

    # Figure 1: GT vs Predicted
    fig1 = plt.figure(figsize=(12, 5), facecolor='white')
    fig1.suptitle(f'{task} / {variation} / {episode_name} - GT vs Predicted\n{success_str} (reward: {max_reward:.2f})',
                  fontsize=11)

    ax1_left = fig1.add_subplot(121, projection='3d')
    plot_trajectory_3d(ax1_left, gt_left, f'Ground Truth ({len(gt_left)} pts)', gt_color,
                       show_waypoints=True, waypoint_size=5)
    if len(pred_left) > 0:
        plot_trajectory_3d(ax1_left, pred_left, f'Predicted ({len(pred_left)} pts)', pred_color,
                           linestyle='--', show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax1_left, 'Left Arm')

    ax1_right = fig1.add_subplot(122, projection='3d')
    plot_trajectory_3d(ax1_right, gt_right, f'Ground Truth ({len(gt_right)} pts)', gt_color,
                       show_waypoints=True, waypoint_size=5)
    if len(pred_right) > 0:
        plot_trajectory_3d(ax1_right, pred_right, f'Predicted ({len(pred_right)} pts)', pred_color,
                           linestyle='--', show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax1_right, 'Right Arm')

    plt.tight_layout()

    # Figure 2: GT vs Executed
    fig2 = plt.figure(figsize=(12, 5), facecolor='white')
    fig2.suptitle(f'{task} / {variation} / {episode_name} - GT vs Executed\n{success_str} (reward: {max_reward:.2f})',
                  fontsize=11)

    ax2_left = fig2.add_subplot(121, projection='3d')
    plot_trajectory_3d(ax2_left, gt_left, f'Ground Truth ({len(gt_left)} pts)', gt_color,
                       show_waypoints=True, waypoint_size=5)
    if len(exec_left) > 0:
        plot_trajectory_3d(ax2_left, exec_left[:, :3], f'Executed ({len(exec_left)} pts)', exec_color,
                           show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax2_left, 'Left Arm')

    ax2_right = fig2.add_subplot(122, projection='3d')
    plot_trajectory_3d(ax2_right, gt_right, f'Ground Truth ({len(gt_right)} pts)', gt_color,
                       show_waypoints=True, waypoint_size=5)
    if len(exec_right) > 0:
        plot_trajectory_3d(ax2_right, exec_right[:, :3], f'Executed ({len(exec_right)} pts)', exec_color,
                           show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax2_right, 'Right Arm')

    plt.tight_layout()

    # Figure 3: Predicted vs Executed
    fig3 = plt.figure(figsize=(12, 5), facecolor='white')
    fig3.suptitle(f'{task} / {variation} / {episode_name} - Predicted vs Executed\n{success_str} (reward: {max_reward:.2f})',
                  fontsize=11)

    ax3_left = fig3.add_subplot(121, projection='3d')
    if len(pred_left) > 0:
        plot_trajectory_3d(ax3_left, pred_left, f'Predicted ({len(pred_left)} pts)', pred_color,
                           linestyle='--', show_waypoints=True, waypoint_size=30)
    if len(exec_left) > 0:
        plot_trajectory_3d(ax3_left, exec_left[:, :3], f'Executed ({len(exec_left)} pts)', exec_color,
                           show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax3_left, 'Left Arm')

    ax3_right = fig3.add_subplot(122, projection='3d')
    if len(pred_right) > 0:
        plot_trajectory_3d(ax3_right, pred_right, f'Predicted ({len(pred_right)} pts)', pred_color,
                           linestyle='--', show_waypoints=True, waypoint_size=30)
    if len(exec_right) > 0:
        plot_trajectory_3d(ax3_right, exec_right[:, :3], f'Executed ({len(exec_right)} pts)', exec_color,
                           show_waypoints=True, waypoint_size=30)
    setup_3d_axis(ax3_right, 'Right Arm')

    plt.tight_layout()

    if show:
        plt.show()

    return fig1, fig2, fig3


def visualize_all_episodes(trajectory_dir, show=True):
    """Visualize all episodes in a trajectory directory one by one."""
    npz_files = sorted(glob.glob(os.path.join(trajectory_dir, '**', '*_trajectories.npz'), recursive=True))

    if not npz_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return

    print(f"Found {len(npz_files)} trajectory files")
    print("Press any key to advance to next episode, or close all windows to exit.")

    for i, npz_path in enumerate(npz_files):
        print(f"\n[{i+1}/{len(npz_files)}] {npz_path}")
        try:
            visualize_episode(npz_path, show=show)
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")


def create_summary_plot(trajectory_dir, show=True):
    """Create a summary plot showing all episodes overlaid."""
    npz_files = sorted(glob.glob(os.path.join(trajectory_dir, '**', '*_trajectories.npz'), recursive=True))

    if not npz_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return

    fig = plt.figure(figsize=(14, 6), facecolor='white')

    ax_left = fig.add_subplot(121, projection='3d')
    ax_right = fig.add_subplot(122, projection='3d')

    success_count = 0
    total_count = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        success = data['success']
        total_count += 1
        if success:
            success_count += 1

        exec_left = data['exec_left']
        exec_right = data['exec_right']

        color = 'green' if success else 'red'
        alpha = 0.4

        if len(exec_left) > 0:
            ax_left.plot(exec_left[:, 0], exec_left[:, 1], exec_left[:, 2],
                        color=color, linewidth=0.5, alpha=alpha)

        if len(exec_right) > 0:
            ax_right.plot(exec_right[:, 0], exec_right[:, 1], exec_right[:, 2],
                         color=color, linewidth=0.5, alpha=alpha)

    # Plot ground truth from first episode as reference
    data = np.load(npz_files[0], allow_pickle=True)
    gt_left = data['gt_left']
    gt_right = data['gt_right']

    ax_left.plot(gt_left[:, 0], gt_left[:, 1], gt_left[:, 2],
                 color='blue', linewidth=2, label='GT (ep0)')
    ax_right.plot(gt_right[:, 0], gt_right[:, 1], gt_right[:, 2],
                  color='blue', linewidth=2, label='GT (ep0)')

    task = os.path.basename(os.path.dirname(os.path.dirname(npz_files[0])))

    fig.suptitle(f'{task} - All Episodes Summary\n'
                 f'Success: {success_count}/{total_count} ({100*success_count/total_count:.1f}%) | '
                 f'Green=Success | Red=Failure | Blue=GT', fontsize=11)

    ax_left.set_title('Left Arm', fontsize=10)
    ax_left.set_xlabel('X', fontsize=8)
    ax_left.set_ylabel('Y', fontsize=8)
    ax_left.set_zlabel('Z', fontsize=8)
    ax_left.legend(fontsize=7)

    ax_right.set_title('Right Arm', fontsize=10)
    ax_right.set_xlabel('X', fontsize=8)
    ax_right.set_ylabel('Y', fontsize=8)
    ax_right.set_zlabel('Z', fontsize=8)
    ax_right.legend(fontsize=7)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved trajectories")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                        help="Directory containing trajectory .npz files")
    parser.add_argument("--single", type=str, default=None,
                        help="Path to single .npz file to visualize")
    parser.add_argument("--summary", action="store_true",
                        help="Create summary plot with all episodes overlaid")

    args = parser.parse_args()

    if args.single:
        visualize_episode(args.single, show=True)
    elif args.summary:
        create_summary_plot(args.trajectory_dir, show=True)
    else:
        visualize_all_episodes(args.trajectory_dir, show=True)
