import cv2
import numpy as np


# input is numpy image array
def opticalflow(img1, img2):

    prvs = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    next = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255
    inst = cv2.optflow.createOptFlow_DeepFlow()
    flow = inst.calc(prvs, next, None)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    mask = np.where(gray > 5, 0, 1)


    return flow , mask