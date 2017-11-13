# -*- coding: utf-8 -*-
from paf import BuildModel
import util

import json
import argparse
import cv2

# model config
param = {'use_gpu': 1,
     'GPUdeviceNumber': 0,
     'modelID': '1',
     'octave': 3,
     'starting_range': 0.8,
     'ending_range': 2.0,
     'scale_search': [0.5, 1, 1.5, 2],
     'thre1': 0.1,
     'thre2': 0.05,
     'thre3': 0.5,
     'min_num': 4,
     'mid_num': 10,
     'crop_ratio': 2.5,
     'bbox_ratio': 0.25
    }
model_params = {'boxsize': 368,
                'padValue': 128,
                'np': '12',
                'stride': 8,
                'part_str': ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho',
                    'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne',
                    'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19]']}


# inference function
def inferene(model, inputs):

    return outputs

# json function
def compute_maps(maps):

    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pose estimation model")
    parser.add_argument("target_path",
            type=str,
            help="target file path")
    parser.add_argument("--weights_path",
            default="./models/model.h5",
            type=str,
            help="weight path")
    parser.add_argument("--mode",
            default="movie",
            choices=["movie", "image"],
            type=str,
            help="estimation mode")
    parser.add_argument("target_path",
            type=str,
            help="target file path")
    args = parser.parse_args()

    print("-----Loading weight-----")
    model = BuildModel(args.weights_path)
    #print(model.summary())
    print("---Done---")

    # if use multiscale estimation
    multiplier = [x * model_params["boxsize"] / oriImg.shape[0] for x in param["scale_search"]]

    # shape is origimg shape and channel is np
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    # part affinity firld の数
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

    scale = multiplier[0]
    # origimgのサイズにscaleをかける
    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 下と右をpadding padはどれだけpaddingしたか imageTo
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])
    # 入力用へ　(bacth, height, width, ch) channel last
    input_img = np.transpose(imageToTest_padded[:,:,:,np.newaxis], (3,0,1,2))
    # predicted [0]:paf predicted[1]heatmap
    output_blobs = model.predict(input_img)
    # (1,h,w,ch) -> (h,w,ch)
    paf = np.squeeze(output_blobs[0])
    heatmap = np.squeeze(output_blobs[1])
