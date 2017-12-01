# -*- coding: utf-8 -*-
import cv2
import json
import math
import scipy
import os
import numpy as np
import matplotlib
import argparse
import pylab as plt
import util
from tqdm import tqdm

model_params = {'boxsize': 368,
                'padValue': 128,
                'np': '12',
                'stride': 8,
                'part_str': ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19]']}

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]



def visualize_points(output, canvas, circlesize):
    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        X = output["points"][model_params["part_str"][i]][0]
        Y = output["points"][model_params["part_str"][i]][1]
        if None in [X, Y]:
            continue
        cv2.circle(canvas, (int(X), int(Y)), circlesize, colors[1], thickness=-1)
    return canvas


def visualize_limb(output, canvas, stickwidth):
    stickwidth = 4
    for i in range(17):
        index = [output['points'][model_params["part_str"][limbSeq[i][0]-1]], output['points'][model_params["part_str"][limbSeq[i][1]-1]]]
        if None in index:
            continue
        cur_canvas = canvas.copy()
        Y = [index[0][0], index[1][0]]
        X = [index[0][1], index[1][1]]
        if None in X or None in Y:
            continue
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[0])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

def load_img(img_path):
    img = cv2.imread(img_path)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize born model")
    parser.add_argument("--target_path",
            default="/mnt/storage/clients/rakuten/Kobo/20170630_3P/01.mp4",
            type=str,
            help="target file path")
    parser.add_argument("--json_dir",
            default="./result/02",
            type=str,
            help="json dirctory")
    parser.add_argument("--json_path",
            type=str,
            help="json path")
    parser.add_argument("--mode",
            default="movie",
            choices=["movie", "image"],
            type=str,
            help="estimation mode")
    parser.add_argument("--handedness",
            default="right",
            choices=["right", "left"],
            type=str,
            help="picher handness")
    parser.add_argument("--stickwidth",
            default=4,
            type=int,
            help="born stickwidth")
    parser.add_argument("--circlesize",
            default=5,
            type=int,
            help="circle size")
    args = parser.parse_args()

    if args.mode == "movie":
        # frame number
        frame_num = 0

        # json list
        json_list = os.listdir(args.json_dir)
        json_list.sort()
        json_name = json_list[0]

        # get json
        json_path = json_list[frame_num]
        with open(args.json_dir+"/"+json_path, "r") as f:
            output = json.load(f)

        # base name
        base = os.path.basename(args.json_dir)
        if os.path.isdir("result/movie/"+base):
            os.mkdir("result/movie/"+base)

        # output path
        output_path = "result/movie/"+base+".mp4"

        # load movie
        cap = cv2.VideoCapture(args.target_path)

        # save movie
        rec = cv2.VideoWriter(output_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                30,
                (output["img_shape"][0], output["img_shape"][1]),
                True)

        cmap = matplotlib.cm.get_cmap("hsv")

        nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(nb_frame)):
            # get frame
            ret, frame = cap.read()
            if ret == False:
                break

            # get json
            json_path = json_list[frame_num]
            with open(args.json_dir+"/"+json_path, "r") as f:
                output = json.load(f)

            canvas = frame.copy()
            canvas, output = util.getGrove(canvas, output, args.handedness)
            canvas = visualize_points(output, canvas, args.circlesize)
            canvas = visualize_limb(output, canvas, args.stickwidth)

            rec.write(canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_num += 1

        cv2.destroyAllWindows()
        rec.release()

    elif mode == "image":
        origimg = load_img(args.target_path)

    else:
        print("wrong mode")

