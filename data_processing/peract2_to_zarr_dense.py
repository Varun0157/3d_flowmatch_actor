"""Convert raw PerAct2 episodes to a dense zarr dataset.

Unlike peract2_to_zarr.py (which only stores keyposes), this script stores the
full joint-space trajectory between each pair of consecutive keyposes. For each
episode segment [keypose_i → keypose_{i+1}], all simulation frames in that
interval are subsampled to exactly max_steps frames. No interpolation is used —
frames are actual recorded simulator states.

Resulting action_joints shape: (N, max_steps, 2, 8)
Observations (rgb/depth/proprio) are still taken at the start of each segment
(the keypose the arm is at when the segment begins).

Usage:
    python data_processing/peract2_to_zarr_dense.py \\
        --root /path/to/peract2_raw \\
        --tgt /path/to/peract2_dense_zarr \\
        --max_steps 50
"""

import argparse
import json
import os
import pickle

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions
)


NCAM = 3
NHAND = 2
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1


def parse_arguments():
    parser = argparse.ArgumentParser()
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/peract2_raw_squash/'),
        ('tgt', str, 'zarr_datasets/peract2_dense/'),
        ('max_steps', int, 15)
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])
    return parser.parse_args()


def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


def subsample_to_length(frames, length):
    """Subsample a list of frame indices to exactly `length`.
    Subsamples uniformly if longer; repeats the last frame if shorter."""
    T = len(frames)
    if T >= length:
        indices = np.round(np.linspace(0, T - 1, length)).astype(int)
        return [frames[i] for i in indices]
    else:
        return frames + [frames[-1]] * (length - T)


def all_tasks_main(split, tasks, root, store_path, max_steps):
    filename = f"{store_path}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr {filename} already exists. Skipping...")
        return

    cameras = ["front", "wrist_left", "wrist_right"]
    task2id = {task: t for t, task in enumerate(tasks)}

    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field, shape=(0,) + shape, chunks=(1,) + shape,
                compressor=compressor, dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create("action", (1, NHAND, 8), "float32")                   # keypose EEF goal
        _create("action_joints", (max_steps, NHAND, 8), "float32")    # dense joint traj
        _create("proprioception_joints", (1, NHAND, 8), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        for task in tasks:
            print(task)
            task_folder = f'{root}/{split}/{task}/all_variations/episodes'
            if not os.path.exists(task_folder):
                print(f"  Skipping (not found): {task_folder}")
                continue
            episodes = sorted(os.listdir(task_folder))
            for ep in tqdm(episodes):
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                key_frames = keypoint_discovery(demo, bimanual=True)
                key_frames.insert(0, 0)
                n_segments = len(key_frames) - 1
                if n_segments == 0:
                    continue

                # RGB observations taken at the start of each segment
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{task_folder}/{ep}/{cam}_rgb/rgb_{_num2id(k)}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ]).transpose(0, 1, 4, 2, 3)

                # Depth observations at segment starts
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in cameras:
                        d = image_to_float_array(Image.open(
                            f"{task_folder}/{ep}/{cam}_depth/depth_{_num2id(k)}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        cam_d.append((near + d * (far - near)).astype(np.float16))
                    depth_list.append(np.stack(cam_d))
                depth = np.stack(depth_list)

                # EEF proprioception history (3 frames) at segment starts
                eef_states = np.stack([np.concatenate([
                    demo[k].left.gripper_pose, [demo[k].left.gripper_open],
                    demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                prop = eef_states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1).reshape(n_segments, 3, NHAND, 8)

                # Keypose EEF actions (goal poses, for reference)
                actions_eef = eef_states[1:].reshape(n_segments, 1, NHAND, 8)

                # Joint state at segment starts
                jnt_states = np.stack([np.concatenate([
                    demo[k].left.joint_positions, [demo[k].left.gripper_open],
                    demo[k].right.joint_positions, [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                prop_jnts = jnt_states[:-1].reshape(n_segments, 1, NHAND, 8)

                # Dense joint trajectories — actual simulator frames, no interpolation
                dense_trajs = []
                for seg_i in range(n_segments):
                    start_f, end_f = key_frames[seg_i], key_frames[seg_i + 1]
                    # Frames from start+1 through end (the arm's motion to next keypose)
                    segment = list(range(start_f + 1, end_f + 1)) or [end_f]
                    segment = subsample_to_length(segment, max_steps)
                    traj = np.array([
                        [
                            np.concatenate([demo[f].left.joint_positions,  [demo[f].left.gripper_open]]),
                            np.concatenate([demo[f].right.joint_positions, [demo[f].right.gripper_open]])
                        ]
                        for f in segment
                    ], dtype=np.float32)  # (max_steps, 2, 8)
                    dense_trajs.append(traj)
                dense_trajs = np.stack(dense_trajs)  # (n_segments, max_steps, 2, 8)

                extrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16) for cam in cameras])
                    for k in key_frames[:-1]
                ])
                intrinsics = np.stack([
                    np.stack([demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16) for cam in cameras])
                    for k in key_frames[:-1]
                ])

                task_id = np.full(n_segments, task2id[task], dtype=np.uint8)
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    variation = np.full(n_segments, int(pickle.load(f)), dtype=np.uint8)

                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions_eef)
                zarr_file['action_joints'].append(dense_trajs)
                zarr_file['proprioception_joints'].append(prop_jnts)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(variation)

    print(f"\nCreated: {filename}")
    with zarr.open(filename, 'r') as check:
        print("Shapes:")
        for k in sorted(check.keys()):
            print(f"  {k}: {check[k].shape}")


if __name__ == "__main__":
    tasks = [
        'bimanual_push_box',
        'bimanual_pick_plate',
    ]
    args = parse_arguments()
    os.makedirs(args.tgt, exist_ok=True)
    for split in ['train', 'val']:
        all_tasks_main(split, tasks, args.root, args.tgt, args.max_steps)
    os.makedirs('instructions/peract2', exist_ok=True)
    instr_dict = store_instructions(args.root, tasks)
    with open('instructions/peract2/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)
