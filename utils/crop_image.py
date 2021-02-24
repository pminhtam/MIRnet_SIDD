import cv2
import numpy as np
from glob import glob
import os
import math
import argparse
def crop_image(img, crop_size):
    h,w,_ = img.shape
    h_crop_size, w_crop_size = crop_size
    num_h = math.ceil(h/h_crop_size)
    num_w = math.ceil(w/w_crop_size)
    dis_h = math.floor((h-h_crop_size)/(num_h-1))
    dis_w = math.floor((w-h_crop_size)/(num_w-1))
    print(h, num_h,num_w)
    list_img_crop = []
    for i in range(num_h):
        for j in range(num_w):
            list_img_crop.append(img[i*dis_h:i*dis_h+h_crop_size,j*dis_w:j*dis_w+w_crop_size,:])
    return list_img_crop


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--image_path', '-i', default='/home/dell/Downloads/noise_raw/0001_NOISY_RAW/', help='path to noise folder image')
    # parser.add_argument('--image_path', '-n', default='/home/dell/Downloads/noise_raw/split/', help='path to noise folder image')
    parser.add_argument('--save_path', '-s', default='/home/dell/Downloads/noise_raw/split/', help='path to gt folder image')
    parser.add_argument('--crop_size', '-c', default=256, type=int, help='Crop size')

    args = parser.parse_args()
    #
    files_ = sorted(glob(os.path.join(args.image_path,'*.jpeg')))
    root_split = args.save_path
    crop_size = (args.crop_size,args.crop_size)

    if not os.path.isdir(root_split):
        os.mkdir(root_split)
    for fi in files_:
        name = fi.split("/")[-1].split(".")[0]
        print(name)
        # noisy_img = cv2.imread(fi+"/NOISY_SRGB_010.PNG")/255.0
        # img = cv2.imread(fi+"/NOISY_SRGB_010.PNG")
        # img = cv2.imread(fi+"/GT_SRGB_010.PNG")
        img = cv2.imread(fi)
        list_img_crop = crop_image(img, crop_size)
        for i in range(len(list_img_crop)):
            cv2.imwrite(os.path.join(root_split , name + str(i)+".jpeg"), list_img_crop[i])
