# -*- coding: utf-8 -*-
import os
import json
import argparse
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy
import math
from tqdm import tqdm

from paf import BuildModel
import util


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

# model paraeters
model_params = {'boxsize': 368,
                'padValue': 128,
                'np': '12',
                'stride': 8,
                'part_str': ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho',
                    'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne',
                    'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19]']}

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]
# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

# load image
def loadimg(img_path):
    origimg = cv2.imread(img_path)

    return origimg

# img -> input tensor
def createinput(img, scale):
    # origimgのサイズにscaleをかける
    imageToTest = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # 下と右をpadding padはどれだけpaddingしたか imageTo
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])

    return imageToTest_padded, pad

# inference function
def inferene(model, imageToTest_padded, pad):
    # 入力用へ (bacth, height, width, ch) channel last
    input_img = np.transpose(imageToTest_padded[:,:,:,np.newaxis], (3,0,1,2))
    # predicted outputs[0]paf and outputs[1]heatmap
    output_blobs = model.predict(input_img)
    # (1,h,w,ch) -> (h,w,ch)
    paf = np.squeeze(output_blobs[0])
    heatmap = np.squeeze(output_blobs[1])
    # resize maps size to padded inpus size
    heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]

    return paf, heatmap

# compute joints
def compute_joint(heatmap, paf, origimg):
    # peak with score and id
    # 関節ごとにまとめられたリスト
    all_peaks = []
    # 検出した関節の個数
    peak_counter = 0

    connection_all = []
    special_k = []
    mid_num = 10

    for part in range(19-1):
        map_ori = heatmap[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map>param['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    for k in range(len(mapIdx)):
        score_mid = paf[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*origimg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    # personごとにまとめる
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    # print ("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return subset, candidate

# intify picher
def create_dict(subset, candidate, oriImg):
    disperision = np.array([])
    for n in range(len(subset)):
        #X_coords = np.array([])
        Y_coords = np.array([])
        for i in range(18):
            if subset[n][i] >= 0:
                #X = candidate[subset[n][i], 0]
                Y = candidate[subset[n][i], 1]
                #X_coords = np.append(coords, X)
                Y_coords = np.append(Y_coords, Y)
        disperision = np.append(disperision,np.var(Y_coords))

    person_id = np.argmax(disperision)

    output = {"img_shape":(oriImg.shape[1], oriImg.shape[0]),
          "points":{}}

    for i in range(18):
        if subset[person_id][i] >= 0:
            X = candidate[subset[person_id][i], 0]
            Y = candidate[subset[person_id][i], 1]
            output["points"][model_params["part_str"][i]] = (X, Y)
        else:
            X = None
            Y = None
            output["points"][model_params["part_str"][i]] = (X, Y)

    return output

# dict to json
def dict2json(output, name):
    json_str = json.dumps(output)
    json_dict = json.loads(json_str)
    # json データの書き込み
    with open(name+'.json', 'w') as f:
        json.dump(json_dict, f, indent=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pose estimation model")
    parser.add_argument("--target_path",
            default="/mnt/storage/clients/rakuten/Kobo/20170630_P/01.mp4",
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
    parser.add_argument("--gpu_num",
            default="0",
            type=str,
            help="gpu_num")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    print("-----Loading weight-----")
    model = BuildModel(args.weights_path)
    #print(model.summary())
    print("---Done---")

    if args.mode == "image":
        # load orignal image
        origimg = loadimg(args.target_path)
        multiplier = [x * model_params["boxsize"] / origimg.shape[0] for x in param["scale_search"]]
        scale = multiplier[0]
        # if use multiscale estimation
        # create inputs with padding
        imageToTest_padded, pad = createinput(origimg, scale)
        # forward inference function
        paf, heatmap = inferene(model, imageToTest_padded, pad)
        heatmap = cv2.resize(heatmap, (origimg.shape[1], origimg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paf, (origimg.shape[1], origimg.shape[0]), interpolation=cv2.INTER_CUBIC)

        subset, candidate = compute_joint(heatmap, paf, origimg)
        output = create_dict(subset, candidate, origimg)
        name = "./result/"+os.path.basename(args.target_path)[:-4]
        dict2json(output, name)

    elif args.mode == "movie":
        name = os.path.basename(args.target_path)[:-4]
        if not os.path.isdir("result/"+name):
            os.mkdir("result/"+name)
        cap = cv2.VideoCapture(args.target_path)
        # get number of frame
        nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_num in tqdm(range(nb_frame)):
            # get frame
            ret, frame = cap.read()
            # cap flag
            if ret == False:
                break
            origimg = frame
            multiplier = [x * model_params["boxsize"] / origimg.shape[0] for x in param["scale_search"]]
            scale = multiplier[0]
            # create inputs with padding
            imageToTest_padded, pad = createinput(origimg, scale)
            # forward inference function
            paf, heatmap = inferene(model, imageToTest_padded, pad)
            heatmap = cv2.resize(heatmap, (origimg.shape[1], origimg.shape[0]), interpolation=cv2.INTER_CUBIC)
            paf = cv2.resize(paf, (origimg.shape[1], origimg.shape[0]), interpolation=cv2.INTER_CUBIC)

            subset, candidate = compute_joint(heatmap, paf, origimg)
            output = create_dict(subset, candidate, origimg)
            output_path = "./result/"+name+"/"+str("%05.f"%frame_num)
            dict2json(output, output_path)
            frame_num += 1

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        print("done")

    else:
        print("wrong mode")

    if args.mode == "movie":
        cap.release()
        cv2.destroyAllWindows()
