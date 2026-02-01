import argparse
import json
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    store_instructions,
    interpolate_trajectory,
    quat_to_euler_np,
    euler_to_quat_np
)
from utils.common_utils import str2bool


NCAM = 3
NHAND = 2
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1
DEFAULT_INTERP_LEN = 50


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/peract2_raw_squash/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/Peract2_zarr/'),
        ('store_trajectory', str2bool, False),
        ('interp_len', int, DEFAULT_INTERP_LEN),
        ('tasks', str, None)  # comma-separated task names, or None for all
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def _interpolate_bimanual(left_traj, right_traj, num_steps):
    """
    Interpolate bimanual trajectories.
    left_traj, right_traj: (T, 8) each - [x, y, z, qx, qy, qz, qw, gripper]
    Returns: (num_steps, 2, 8)
    """
    def _interp_single(traj):
        # Convert quaternion to euler for smooth interpolation
        traj_euler = np.concatenate((
            traj[:, :3],
            quat_to_euler_np(traj[:, 3:7]),
            traj[:, 7:]
        ), axis=1)
        # Interpolate
        traj_interp = interpolate_trajectory(traj_euler, num_steps)
        # Convert back to quaternion
        traj_quat = np.concatenate((
            traj_interp[:, :3],
            euler_to_quat_np(traj_interp[:, 3:6]),
            traj_interp[:, 6:]
        ), axis=1)
        return traj_quat

    left_interp = _interp_single(left_traj)
    right_interp = _interp_single(right_traj)
    return np.stack([left_interp, right_interp], axis=1)


def all_tasks_main(split, tasks, store_trajectory=False, interp_len=DEFAULT_INTERP_LEN):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    cameras = ["front", "wrist_left", "wrist_right"]
    task2id = {task: t for t, task in enumerate(tasks)}

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        traj_len = interp_len if store_trajectory else 1
        _create("action", (traj_len, NHAND, 8), "float32")
        _create("proprioception_joints", (1, NHAND, 8), "float32")
        _create("action_joints", (1, NHAND, 8), "float32")
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        for task in tasks:
            print(task)
            task_folder = f'{ROOT}/{split}/{task}/all_variations/episodes'
            episodes = sorted(os.listdir(task_folder))
            for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{task_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                # Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=True)
                key_frames.insert(0, 0)

                # Loop through keyposes and store:
                # RGB (keyframes, cameras, 3, 256, 256)
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{task_folder}/{ep}/{cam}_rgb/rgb_{_num2id(k)}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])
                rgb = rgb.transpose(0, 1, 4, 2, 3)

                # Depth (keyframes, cameras, 256, 256)
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in cameras:
                        depth = image_to_float_array(Image.open(
                            f"{task_folder}/{ep}/{cam}_depth/depth_{_num2id(k)}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        depth = near + depth * (far - near)
                        cam_d.append(depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                depth = np.stack(depth_list)

                # Proprioception (keyframes, 3, 2, 8)
                states = np.stack([np.concatenate([
                    demo[k].left.gripper_pose, [demo[k].left.gripper_open],
                    demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                # Store current eef pose as well as two previous ones
                prop = states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)

                # Action (keyframes, traj_len, 2, 8)
                if not store_trajectory:
                    actions = states[1:].reshape(len(states[1:]), 1, NHAND, 8)
                else:
                    # Get all states from demo (not just keyposes)
                    all_left = np.stack([np.concatenate([
                        demo[k].left.gripper_pose, [demo[k].left.gripper_open]
                    ]) for k in range(len(demo))]).astype(np.float32)
                    all_right = np.stack([np.concatenate([
                        demo[k].right.gripper_pose, [demo[k].right.gripper_open]
                    ]) for k in range(len(demo))]).astype(np.float32)
                    # Interpolate between each keypose pair
                    actions = np.stack([
                        _interpolate_bimanual(
                            all_left[prev:next_ + 1],
                            all_right[prev:next_ + 1],
                            interp_len
                        )
                        for prev, next_ in zip(key_frames[:-1], key_frames[1:])
                    ])

                # Proprioception in joints (keyframes, 3, 2, 8)
                states = np.stack([np.concatenate([
                    demo[k].left.joint_positions,
                    [demo[k].left.gripper_open],
                    demo[k].right.joint_positions,
                    [demo[k].right.gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                prop_jnts = states[:-1].reshape(len(states[:-1]), 1, NHAND, 8)

                # Action in joints (keyframes, 1, 2, 8)
                actions_jnts = states[1:].reshape(len(states[1:]), 1, NHAND, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                extrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                intrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Task id (keyframes,)
                task_id = np.array([task2id[task]] * len(key_frames[:-1]))
                task_id = task_id.astype(np.uint8)

                # Variation (keyframes,)
                with open(f"{task_folder}/{ep}/variation_number.pkl", 'rb') as f:
                    var_ = pickle.load(f)
                var_ = np.array([int(var_)] * len(key_frames[:-1]))
                var_ = var_.astype(np.uint8)

                # Write
                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['proprioception_joints'].append(prop_jnts)
                zarr_file['action_joints'].append(actions_jnts)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(var_)


def _num2id(int_):
    str_ = str(int_)
    return '0' * (4 - len(str_)) + str_


if __name__ == "__main__":
    ALL_TASKS = [
        'bimanual_push_box',
        'bimanual_lift_ball',
        'bimanual_dual_push_buttons',
        'bimanual_pick_plate',
        'bimanual_put_item_in_drawer',
        'bimanual_put_bottle_in_fridge',
        'bimanual_handover_item',
        'bimanual_pick_laptop',
        'bimanual_straighten_rope',
        'bimanual_sweep_to_dustpan',
        'bimanual_lift_tray',
        'bimanual_handover_item_easy',
        'bimanual_take_tray_out_of_oven'
    ]
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Filter tasks if specified
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        tasks = ALL_TASKS
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks, args.store_trajectory, args.interp_len)
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/peract2', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks)
    with open('instructions/peract2/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)
