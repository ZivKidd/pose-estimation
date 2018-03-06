# -*- coding: utf-8 -*-
import requests
from xml.etree.ElementTree import Element, SubElement, Comment, tostring, fromstring
from xml.dom import minidom
from datetime import datetime
from datetime import timedelta
import cv2
import json
from tqdm import tqdm

from pose_estimation import *


def get_xml(query):
    BASE_URL = "http://reoapi-961460383.flb.i1.fusioncom.jp/api/game_ball_score/game_ball_score.xml"
    response = requests.get(url=BASE_URL, params=query)
    root = fromstring(response.text.encode('utf-8'))
    return root


def predict_and_save(model, cap, play_number, save_dir="20171008E-H-1P", seconds=8):
    base_dir = "./" + save_dir
    if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
    for frame_num in tqdm(range(30 * seconds)):
        # get frame
        flag, frame = cap.read()
        if frame is None:
            print("frame is None")
        # cap flag
        if not flag:
            break
        multiplier = [x * model_params["boxsize"] / frame.shape[0] for x in param["scale_search"]]
        scale = multiplier[0]
        # create inputs with padding
        imageToTest_padded, pad = createinput(frame, scale)
        # forward inference function
        paf, heatmap = inferene(model, imageToTest_padded, pad)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = cv2.resize(paf, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        subset, candidate = compute_joint(heatmap, paf, frame)
        output = create_dict(subset, candidate, frame)

        # get grove area and position
        _, output = utils.getGrove(frame, output, handedness="right")

        # dict to json
        output_dir = base_dir + "/" + str("%06.f"%(play_number))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_path =  output_dir + "/" + str("%06.f"%(frame_num+1)) + ".json"
        json_dict = dict2json(output)

        # save json
        with open(output_path, 'w') as f:
            json.dump(json_dict, f, indent=3)


def main(query, movie_path, seconds):
    # 楽天APIのレスポンスの取得
    root = get_xml(query)

    # 試合開始時間の取得
    start_time = root.find(".//GameStartAt").text
    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

    timestamps = [datetime.strptime(e.text[:-6], '%Y-%m-%d %H:%M:%S') - timedelta(hours = 9) for e in root.getiterator("PlayAt")]

    # load movie
    cap = cv2.VideoCapture(movie_path)

    # 予測モデルの読み込み
    model = BuildModel("./models/model.h5")

    # 保存先ディクトリの名前
    save_dir, _ = os.path.splitext(os.path.basename(movie_path))

    for i, v in tqdm(enumerate(timestamps)):
        start_frame_num = (v - start_time).seconds * 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
        predict_and_save(model, cap, play_number=i, save_dir=save_dir, seconds=seconds)


if __name__ == "__main__":
    # query
    query = {
            "game_id": "8811",
            "api_key": "60zbHWim1RoZJuLra4$D308xCAlkhsgn"}
    # 対象動画(1試合分)へのpath
    movie_path = "/mnt/storage/clients/rakuten/gameid_8811/20171008E-H-1P.mpg"
    # ピッチング動作の継続時間
    seconds = 8

    # gpuの指定 cpuを使う場合: ""
    gpu_num = ""
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

    # メイン
    main(query, movie_path, seconds)
