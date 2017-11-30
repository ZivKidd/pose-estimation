# -*- coding: utf-8 -*-
import numpy as np
import cv2
from io import StringIO
import PIL.Image
from IPython.display import Image, display

# RGBimg to GBRimg
def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def showmap(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5
    return c

# gray_img to color_img
def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

# img_padded is padded img and pad is num of padding pixel
def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def getGrove(image, output, handedness="right"):
    # handedness
    if handedness == "right":
        hand = "Lwri"
    else:
        hand = "Rwri"

    # crop near hand
    if output["points"][hand][0] == None or output["points"][hand][1] == None:
        return image, None, 0

    x = int(output["points"][hand][0])
    y = int(output["points"][hand][1])
    xmin = x-100
    ymin = y-50
    xmax = x
    ymax = y+70
    croped = image[ymin:ymax, xmin:xmax].copy()

    # グレースケール
    gray = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)

    # otsuの2値化
    thresh,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # モルフォロジーによる ノイズ消去
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=1)

    #  オブジェクトと背景の距離変換から全景の取得
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    # 背景でも前景でもない部分の取得
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # 領域のラベリング
    ret, markers = cv2.connectedComponents(sure_fg)
    # 背景:0 -> 1 , オブジェクト:1~ -> 2~
    markers = markers+1
    # unknown:0 , 背景:1,  オブジェクト:2~
    markers[unknown==255] = 0

    # watarshed
    markers = cv2.watershed(croped,markers)

    # max_id:  最大の領域のオブジェクト, nb_pixel: 最大領域のピクセル数
    max_id = np.unique(markers,return_counts=True)[1][2:].argmax()+2

    # 元の画像でのグローブのインデックス
    index = np.where(markers == max_id)
    index[0][:] += xmin
    index[1][:] += ymin

    # 最大領域のpixel数
    nb_pixel = np.unique(markers,return_counts=True)[1][max_id]
    croped[markers == max_id] = [0, 0, 255]
    image[ymin:ymax, xmin:xmax] = croped

    return image, index, nb_pixel
