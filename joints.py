# -*- coding: utf-8 -*-
import argparse
import json
import os
from tqdm import tqdm
from computeAngle import computeJoint


# get angle function
def angle(json_dir, joint_list):
    # consts
    result = {"joints": joint_list}
    frame_num = 0

    # get json list
    json_list = os.listdir(json_dir)
    json_list.sort()
    nb_frame = len(json_list)

    # base name
    base = os.path.basename(json_dir)
    if os.path.isdir("result/angle/" + base):
        os.mkdir("result/angle/" + base)

    # output path
    output_path = "result/angle/" + base + ".json"

    for frame_num in tqdm(range(nb_frame)):
        # get json
        json_path = json_list[frame_num]
        with open(json_dir + "/" + json_path, "r") as f:
            output = json.load(f)

        computer = computeJoint(output, joint_list)
        angles = computer.angle()
        result[str(frame_num)] = angles
    # save json
    with open(output_path, "w") as f:
        json.dump(result, f, indent=3)


# get distance function
def distance(json_dir, joint_list):
    # consts
    result = {"joints": joint_list[0:2]}
    frame_num = 0

    # get json list
    json_list = os.listdir(json_dir)
    json_list.sort()
    nb_frame = len(json_list)

    # base name
    base = os.path.basename(json_dir)
    if os.path.isdir("result/distance/" + base):
        os.mkdir("result/distance/" + base)

    # output path
    output_path = "result/distance/" + base + ".json"

    for frame_num in tqdm(range(nb_frame)):
        # get json
        json_path = json_list[frame_num]
        with open(json_dir + "/" + json_path, "r") as f:
            output = json.load(f)

        computer = computeJoint(output, joint_list)
        dist = computer.distance()
        result[str(frame_num)] = dist
    # save json
    with open(output_path, "w") as f:
        json.dump(result, f, indent=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pose estimation model")
    parser.add_argument(
            "--json_dir",
            "-j",
            default="./result/01",
            type=str,
            help="target json dir path")
    parser.add_argument(
            "--joint_list",
            "-l",
            default=["Rsho", "Rhip", "nose"],
            type=list,
            help="target joint list")
    parser.add_argument(
            "--mode",
            "-m",
            default="angle",
            type=str,
            help="type")
    args = parser.parse_args()

    if args.mode == "angle":
        angle(args.json_dir, args.joint_list)
    elif args.mode == "distance":
        distance(args.json_dir, args.joint_list)
