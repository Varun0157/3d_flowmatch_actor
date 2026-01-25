"""Visualize saved trajectories using Plotly 3D plots."""

import argparse
import os
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_gradient_colors(n_points, colorscale_name):
    """Create a list of colors with gradient based on timestep."""
    import plotly.colors as pc

    if colorscale_name == "blues":
        colors = pc.sequential.Blues
    elif colorscale_name == "reds":
        colors = pc.sequential.Reds
    elif colorscale_name == "greens":
        colors = pc.sequential.Greens
    elif colorscale_name == "oranges":
        colors = pc.sequential.Oranges
    else:
        colors = pc.sequential.Viridis

    result = []
    for i in range(n_points):
        t = i / max(n_points - 1, 1)
        idx = int(t * (len(colors) - 1))
        result.append(colors[idx])

    return result


def plot_trajectory(fig, traj, name, color_base, row, col,
                    show_legend=True, marker_size=4, line_width=2, dash=None):
    """Add a trajectory to the plot with time gradient."""
    if len(traj) == 0:
        return

    n_points = len(traj)
    colors = create_gradient_colors(n_points, color_base)

    # Line trace
    fig.add_trace(
        go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            name=name,
            line=dict(
                color=colors[len(colors)//2],
                width=line_width,
                dash=dash
            ),
            marker=dict(
                size=marker_size,
                color=list(range(n_points)),
                colorscale=color_base.capitalize(),
                showscale=False,
            ),
            showlegend=show_legend,
            legendgroup=name,
        ),
        row=row, col=col
    )

    # Start marker
    fig.add_trace(
        go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            name=f'{name} (start)',
            marker=dict(
                size=10,
                color=colors[0],
                line=dict(color='black', width=2),
                symbol='circle',
            ),
            showlegend=False,
            legendgroup=name,
        ),
        row=row, col=col
    )

    # End marker
    fig.add_trace(
        go.Scatter3d(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            z=[traj[-1, 2]],
            mode='markers',
            name=f'{name} (end)',
            marker=dict(
                size=12,
                color=colors[-1],
                line=dict(color='black', width=2),
                symbol='diamond',
            ),
            showlegend=False,
            legendgroup=name,
        ),
        row=row, col=col
    )


def visualize_episode(npz_path, output_path=None, show=False):
    """Visualize trajectories from a single episode with 3x2 pairwise comparison grid."""
    data = np.load(npz_path, allow_pickle=True)

    gt_left = data['gt_left']
    gt_right = data['gt_right']
    pred_left = data['pred_left']
    pred_right = data['pred_right']
    exec_left = data['exec_left']
    exec_right = data['exec_right']
    success = data['success']
    max_reward = data['max_reward']

    # Create 3x2 subplot grid
    # Rows: GT vs Pred, GT vs Exec, Pred vs Exec
    # Cols: Left arm, Right arm
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Left: GT vs Predicted', 'Right: GT vs Predicted',
            'Left: GT vs Executed', 'Right: GT vs Executed',
            'Left: Predicted vs Executed', 'Right: Predicted vs Executed'
        ),
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}]
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )

    # Row 1: GT vs Predicted
    plot_trajectory(fig, gt_left, 'Ground Truth', 'blues', 1, 1)
    if len(pred_left) > 0:
        plot_trajectory(fig, pred_left, 'Predicted', 'reds', 1, 1, dash='dash')

    plot_trajectory(fig, gt_right, 'Ground Truth', 'blues', 1, 2, show_legend=False)
    if len(pred_right) > 0:
        plot_trajectory(fig, pred_right, 'Predicted', 'reds', 1, 2, show_legend=False, dash='dash')

    # Row 2: GT vs Executed
    plot_trajectory(fig, gt_left, 'Ground Truth', 'blues', 2, 1, show_legend=False)
    if len(exec_left) > 0:
        plot_trajectory(fig, exec_left[:, :7], 'Executed', 'greens', 2, 1)

    plot_trajectory(fig, gt_right, 'Ground Truth', 'blues', 2, 2, show_legend=False)
    if len(exec_right) > 0:
        plot_trajectory(fig, exec_right[:, :7], 'Executed', 'greens', 2, 2, show_legend=False)

    # Row 3: Predicted vs Executed
    if len(pred_left) > 0:
        plot_trajectory(fig, pred_left, 'Predicted', 'reds', 3, 1, show_legend=False, dash='dash')
    if len(exec_left) > 0:
        plot_trajectory(fig, exec_left[:, :7], 'Executed', 'greens', 3, 1, show_legend=False)

    if len(pred_right) > 0:
        plot_trajectory(fig, pred_right, 'Predicted', 'reds', 3, 2, show_legend=False, dash='dash')
    if len(exec_right) > 0:
        plot_trajectory(fig, exec_right[:, :7], 'Executed', 'greens', 3, 2, show_legend=False)

    # Extract episode info from path
    episode_name = os.path.basename(npz_path).replace('_trajectories.npz', '')
    variation = os.path.basename(os.path.dirname(npz_path))
    task = os.path.basename(os.path.dirname(os.path.dirname(npz_path)))

    success_str = "SUCCESS" if success else "FAILURE"
    title_color = "green" if success else "red"

    # Update layout with white background
    fig.update_layout(
        title=dict(
            text=f'{task} / {variation} / {episode_name}<br>'
                 f'<span style="font-size:14px; color:{title_color}">Result: {success_str} (max_reward: {max_reward:.2f})</span>',
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        width=1200,
        height=1200,
        margin=dict(l=0, r=0, t=80, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Update all 3D scenes with white background
    scene_settings = dict(
        xaxis=dict(title='X', backgroundcolor='white', gridcolor='lightgray'),
        yaxis=dict(title='Y', backgroundcolor='white', gridcolor='lightgray'),
        zaxis=dict(title='Z', backgroundcolor='white', gridcolor='lightgray'),
        bgcolor='white',
        aspectmode='data'
    )

    fig.update_layout(
        scene=scene_settings,
        scene2=scene_settings,
        scene3=scene_settings,
        scene4=scene_settings,
        scene5=scene_settings,
        scene6=scene_settings
    )

    if output_path:
        fig.write_html(output_path)
        print(f"Saved plot to {output_path}")

    if show:
        fig.show()

    return fig


def visualize_all_episodes(trajectory_dir, output_dir=None, show=False):
    """Visualize all episodes in a trajectory directory."""
    npz_files = sorted(glob.glob(os.path.join(trajectory_dir, '**', '*_trajectories.npz'), recursive=True))

    if not npz_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return

    print(f"Found {len(npz_files)} trajectory files")

    for npz_path in npz_files:
        if output_dir:
            rel_path = os.path.relpath(npz_path, trajectory_dir)
            output_path = os.path.join(output_dir, rel_path.replace('.npz', '.html'))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = npz_path.replace('.npz', '.html')

        try:
            visualize_episode(npz_path, output_path, show=show)
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")


def create_summary_plot(trajectory_dir, output_path=None, show=False):
    """Create a summary plot showing all episodes overlaid."""
    npz_files = sorted(glob.glob(os.path.join(trajectory_dir, '**', '*_trajectories.npz'), recursive=True))

    if not npz_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Left Arm (All Episodes)', 'Right Arm (All Episodes)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )

    success_count = 0
    total_count = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        success = data['success']
        total_count += 1
        if success:
            success_count += 1

        opacity = 0.6
        exec_left = data['exec_left']
        exec_right = data['exec_right']
        episode_name = os.path.basename(npz_path).replace('_trajectories.npz', '')

        if len(exec_left) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=exec_left[:, 0],
                    y=exec_left[:, 1],
                    z=exec_left[:, 2],
                    mode='lines',
                    name=episode_name,
                    line=dict(
                        color='green' if success else 'red',
                        width=2,
                    ),
                    opacity=opacity,
                    showlegend=(i < 10),
                    legendgroup=episode_name,
                ),
                row=1, col=1
            )

        if len(exec_right) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=exec_right[:, 0],
                    y=exec_right[:, 1],
                    z=exec_right[:, 2],
                    mode='lines',
                    name=episode_name,
                    line=dict(
                        color='green' if success else 'red',
                        width=2,
                    ),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=episode_name,
                ),
                row=1, col=2
            )

    # Plot ground truth from first episode as reference
    data = np.load(npz_files[0], allow_pickle=True)
    gt_left = data['gt_left']
    gt_right = data['gt_right']

    fig.add_trace(
        go.Scatter3d(
            x=gt_left[:, 0],
            y=gt_left[:, 1],
            z=gt_left[:, 2],
            mode='lines',
            name='Ground Truth (ep0)',
            line=dict(color='blue', width=4),
            showlegend=True,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=gt_right[:, 0],
            y=gt_right[:, 1],
            z=gt_right[:, 2],
            mode='lines',
            name='Ground Truth (ep0)',
            line=dict(color='blue', width=4),
            showlegend=False,
        ),
        row=1, col=2
    )

    task = os.path.basename(os.path.dirname(os.path.dirname(npz_files[0])))

    fig.update_layout(
        title=dict(
            text=f'{task} - All Episodes Summary<br>'
                 f'<span style="font-size:14px">Success Rate: {success_count}/{total_count} '
                 f'({100*success_count/total_count:.1f}%) | '
                 f'<span style="color:green">Green=Success</span> | '
                 f'<span style="color:red">Red=Failure</span> | '
                 f'<span style="color:blue">Blue=GT</span></span>',
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        width=1400,
        height=700,
        margin=dict(l=0, r=0, t=100, b=0),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    scene_settings = dict(
        xaxis=dict(title='X', backgroundcolor='white', gridcolor='lightgray'),
        yaxis=dict(title='Y', backgroundcolor='white', gridcolor='lightgray'),
        zaxis=dict(title='Z', backgroundcolor='white', gridcolor='lightgray'),
        bgcolor='white',
        aspectmode='data'
    )
    fig.update_layout(scene=scene_settings, scene2=scene_settings)

    if output_path:
        fig.write_html(output_path)
        print(f"Saved summary plot to {output_path}")

    if show:
        fig.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved trajectories")
    parser.add_argument("--trajectory_dir", type=str, required=True,
                        help="Directory containing trajectory .npz files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for HTML plots (default: same as input)")
    parser.add_argument("--single", type=str, default=None,
                        help="Path to single .npz file to visualize")
    parser.add_argument("--summary", action="store_true",
                        help="Create summary plot with all episodes overlaid")
    parser.add_argument("--show", action="store_true",
                        help="Show plots in browser")

    args = parser.parse_args()

    if args.single:
        output = args.single.replace('.npz', '.html') if not args.output_dir else \
                 os.path.join(args.output_dir, os.path.basename(args.single).replace('.npz', '.html'))
        visualize_episode(args.single, output, show=args.show)
    elif args.summary:
        output = os.path.join(args.output_dir or args.trajectory_dir, 'summary.html')
        create_summary_plot(args.trajectory_dir, output, show=args.show)
    else:
        visualize_all_episodes(args.trajectory_dir, args.output_dir, show=args.show)
