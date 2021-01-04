#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-18 10:26:59

import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image

parser = argparse.ArgumentParser(prog='SIDD Train dataset Generation')
# The orignal SIDD images: /ssd1t/SIDD/
parser.add_argument('--data_dir', default=None, type=str, metavar='PATH',
                                      help="path to save the training set of SIDD, (default: None)")
args = parser.parse_args()

path_all_noisy = glob(os.path.join(args.data_dir, '**/*NOISY*.PNG'), recursive=True)
path_all_noisy = sorted(path_all_noisy)
path_all_gt = [x.replace('NOISY', 'GT') for x in path_all_noisy]
print('Number of big images: {:d}'.format(len(path_all_gt)))

print('Training: Split the original images to small ones!')

pch_size = 512
stride = 512-128
num_patch = 0
C = 3

parent_dir = '/content/drive/MyDrive/New SIDD/crop medium'
import shutil
if os.path.exists(parent_dir):
  shutil.rmtree(parent_dir)
os.mkdir(parent_dir)


for ii in tqdm(range(len(path_all_gt))):
  try:
      if (ii+1) % 10 == 0:
          print('    The {:d} original images'.format(ii+1))
      im_noisy_int8 = cv2.imread(path_all_noisy[ii])[...,::-1]
      H, W, _ = im_noisy_int8.shape
      im_gt_int8 = cv2.imread(path_all_gt[ii])[...,::-1]
      ind_H = list(range(0, H-pch_size+1, stride))
      if ind_H[-1] < H-pch_size:
          ind_H.append(H-pch_size)
      ind_W = list(range(0, W-pch_size+1, stride))
      if ind_W[-1] < W-pch_size:
          ind_W.append(W-pch_size)
      for start_H in ind_H:
          for start_W in ind_W:
              pch_noisy = im_noisy_int8[start_H:start_H+pch_size, start_W:start_W+pch_size][...,::-1]
              pch_gt = im_gt_int8[start_H:start_H+pch_size, start_W:start_W+pch_size][...,::-1]
              # pch_imgs = np.concatenate((pch_noisy, pch_gt), axis=2)
              # h5_file.create_dataset(name=str(num_patch), shape=pch_imgs.shape,
              #                                                 dtype=pch_imgs.dtype, data=pch_imgs)
              folder_path = os.path.join(parent_dir, str(num_patch)) 
              os.mkdir(folder_path) 
              cv2.imwrite(os.path.join(folder_path, 'Noisy/'+ str(num_patch)+'.png'), pch_noisy)
              cv2.imwrite(os.path.join(folder_path, 'Clean/'+ str(num_patch)+'.png'), pch_gt)
              num_patch += 1
  except:
    print(path_all_noisy[ii])


        
print('Total {:d} small images in training set'.format(num_patch))
print('Finish!\n')


