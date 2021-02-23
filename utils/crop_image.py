import cv2
import numpy as np
from glob import glob
import os
import math
def crop_image(img, crop_size):
    h,w,_ = img.shape
    h_crop_size, w_crop_size = crop_size
    num_h = math.ceil(h/h_crop_size)
    num_w = math.ceil(w/w_crop_size)
    dis_h = math.floor((h-h_crop_size)/(num_h-1))
    dis_w = math.floor((w-h_crop_size)/(num_w-1))
    print(h, num_h,dis_h)
    list_img_crop = []
    for i in range(num_h):
        for j in range(num_w):
            list_img_crop.append(img[i*dis_h:i*dis_h+h_crop_size,j*dis_w:j*dis_w+w_crop_size,:])
    return list_img_crop

files_ = sorted(glob('train/denoised/*.jpeg'))
root_split = "Clean/"
if not os.path.isdir(root_split):
    os.mkdir(root_split)
for fi in files_:
    name = fi.split("/")[-1]
    print(name)
    # noisy_img = cv2.imread(fi+"/NOISY_SRGB_010.PNG")/255.0
    # img = cv2.imread(fi+"/NOISY_SRGB_010.PNG")
    # img = cv2.imread(fi+"/GT_SRGB_010.PNG")
    img = cv2.imread(fi)
    crop_size = (1024,1024)
    list_img_crop = crop_image(img, crop_size)
    if not os.path.exists(root_split + name):
        os.mkdir(root_split+name)
    for i in range(len(list_img_crop)):
        cv2.imwrite(root_split + name + str(i)+".jpeg", list_img_crop[i])
