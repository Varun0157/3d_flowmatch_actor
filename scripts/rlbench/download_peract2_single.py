"""Download a single PerAct2 task for testing."""

import argparse
import os
import subprocess


LINK = 'https://dataset.cs.washington.edu/fox/bimanual/image_size_256'

VALID_TASKS = [
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=VALID_TASKS,
                        help='Task name to download')
    parser.add_argument('--root', type=str, default='peract2_raw',
                        help='Root directory to store raw data')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated splits to download (train,val)')
    return parser.parse_args()


def download_task(task, root, splits):
    for split in splits:
        split_dir = f'{root}/{split}'
        os.makedirs(split_dir, exist_ok=True)

        task_dir = f'{split_dir}/{task}'
        if os.path.exists(task_dir):
            print(f"Skipping {task}/{split} - already exists")
            continue

        squashfs_file = f'{task}.{split}.squashfs'
        print(f"Downloading {squashfs_file}...")

        subprocess.run(
            f"wget --no-check-certificate {LINK}/{squashfs_file}",
            shell=True, check=True, cwd=split_dir
        )

        print(f"Extracting {squashfs_file}...")
        subprocess.run(
            f"unsquashfs {squashfs_file}",
            shell=True, check=True, cwd=split_dir
        )

        subprocess.run(
            f"mv squashfs-root {task}",
            shell=True, check=True, cwd=split_dir
        )

        subprocess.run(
            f"rm {squashfs_file}",
            shell=True, check=True, cwd=split_dir
        )

        print(f"Done: {task}/{split}")


if __name__ == "__main__":
    args = parse_arguments()
    splits = [s.strip() for s in args.splits.split(',')]
    download_task(args.task, args.root, splits)
    print(f"\nDownload complete. Now package with:")
    print(f"  python data_processing/peract2_to_zarr.py \\")
    print(f"      --root {args.root}/ \\")
    print(f"      --tgt zarr_datasets/peract2_dense/ \\")
    print(f"      --store_trajectory true")
