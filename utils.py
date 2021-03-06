# -*- coding: utf-8 -*-
import numpy as np
import cv2
from io import StringIO
import PIL.Image
import math


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

def computeGrove(index, nb_pixel, output):
    if output["joints"]["Rhip"] == None or output["joints"]["Lhip"] == None:
        output["grove"] = {"pos":None, "area":None}
        return output
    elif len(index[0]) == 0 or len(index[1]) == 0:
        output["grove"] = {"pos":None, "area":None}
        return output
    elif nb_pixel is None:
        output["grove"] = {"pos":None, "area":None}
        return output
    else:
        # compute grove relative coords
        #pose_center = [int((p1+p2)/2) for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["Lhip"])]
        grove_center = [int((index[0].min()+index[0].max())/2), int((index[1].min()+index[1].max())/2)]
        #pos = (round(grove_center[0]/pose_center[0], 3), round(grove_center[1]/pose_center[1], 3))
        pos = (grove_center[0], grove_center[1])

        # compute grove relative area
        # radius = np.linalg.norm([abs(p1-p2) for (p1, p2) in zip(output["joints"]["Rhip"], output["joints"]["Lhip"])])/2
        # area = round(nb_pixel/int(radius**2*math.pi), 3)
        area = int(nb_pixel)

        output["grove"] = {"pos":pos, "area":area}

        return output

def getGrove(image, output, handedness="right"):
    # handedness
    if handedness == "right":
        hand = "Lwri"
    else:
        hand = "Rwri"

    # crop near hand
    if output["joints"][hand][0] == None or output["joints"][hand][1] == None:
        output["grove"] = {"pos":None, "area":None}
        return image, output

    if not image.shape[2] != 3:
        return image, ouput

    x = int(output["joints"][hand][0])
    y = int(output["joints"][hand][1])
    xmin = x-100
    ymin = y-50
    xmax = x+50
    ymax = y+70
    croped = image[ymin:ymax, xmin:xmax].copy()

    # $B%0%l!<%9%1!<%k(B
    gray = cv2.cvtColor(croped, cv2.COLOR_BGR2GRAY)

    # otsu$B$N(B2$BCM2=(B
    thresh,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # $B%b%k%U%)%m%8!<$K$h$k(B $B%N%$%:>C5n(B
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=1)

    #  $B%*%V%8%'%/%H$HGX7J$N5wN%JQ49$+$iA47J$N<hF@(B
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    # $BGX7J$G$bA07J$G$b$J$$ItJ,$N<hF@(B
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # $BNN0h$N%i%Y%j%s%0(B
    ret, markers = cv2.connectedComponents(sure_fg)
    # $BGX7J(B:0 -> 1 , $B%*%V%8%'%/%H(B:1~ -> 2~
    markers = markers+1
    # unknown:0 , $BGX7J(B:1,  $B%*%V%8%'%/%H(B:2~
    markers[unknown==255] = 0

    # watarshed
    markers = cv2.watershed(croped,markers)

    # max_id:  $B:GBg$NNN0h$N%*%V%8%'%/%H(B, nb_pixel: $B:GBgNN0h$N%T%/%;%k?t(B
    max_id = np.unique(markers,return_counts=True)[1][2:].argmax()+2

    # $B85$N2hA|$G$N%0%m!<%V$N%$%s%G%C%/%9(B
    index = np.where(markers == max_id)
    index[0][:] += xmin
    index[1][:] += ymin

    # $B:GBgNN0h$N(Bpixel$B?t(B
    nb_pixel = np.unique(markers,return_counts=True)[1][max_id]
    croped[markers == max_id] = [0, 0, 255]
    image[ymin:ymax, xmin:xmax] = croped

    # $B9x$N4p=`$H$7$?%0%m!<%V$NLL@Q$H0LCV(B
    output = computeGrove(index, nb_pixel, output)

    return image, output
