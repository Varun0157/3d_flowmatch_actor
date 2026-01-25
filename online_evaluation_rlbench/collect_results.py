import argparse
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser("Parse arguments for main.py")
    parser.add_argument('--folder', type=str)

    return parser.parse_args()


args = parse_arguments()
FOLDER = args.folder

sum_ = 0
tasks = sorted(os.listdir(FOLDER))
tasks = [t for t in tasks if t != 'trajectories']  # skip trajectory save folder
results = []
for folder in tasks:
    with open(f'{FOLDER}/{folder}/eval.json') as fid:
        res = 100 * json.load(fid)[folder]["mean"]
    results.append(res)
    print(folder, res)
    sum_ += res
print(f'Mean on {len(tasks)} tasks', sum_ / len(tasks))
