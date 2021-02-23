import torch
import argparse
from model.MIRNet import MIRNet_DGF
from torch.utils.data import DataLoader
# import h5py
from data.data_provider import SingleLoader
from torchsummary import summary
from utils.metric import calculate_psnr,calculate_ssim
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
import math
from PIL import Image
import glob
import time
import scipy.io
from data.data_provider import pixel_unshuffle

# from torchsummary import summary

def load_data(image_noise,burst_length):
    image_noise_hr = image_noise
    upscale_factor = int(math.sqrt(burst_length))
    image_noise = pixel_unshuffle(image_noise, upscale_factor)
    while len(image_noise) < burst_length:
        image_noise = torch.cat((image_noise,image_noise[-2:-1]),dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop,image_noise_hr.unsqueeze(0)
def test(args):
    model = MIRNet_DGF()

    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['BenchmarkNoisyBlocksSrgb']
    mat_re = np.zeros_like(all_noisy_imgs)
    i_imgs,i_blocks, _,_,_ = all_noisy_imgs.shape

    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block]))
            image_noise, image_noise_hr = load_data(noise, args.burst_length)
            image_noise_hr = image_noise_hr.to(device)
            burst_noise = image_noise.to(device)
            begin = time.time()
            _, pred = model(burst_noise,image_noise_hr)
            pred = pred.detach().cpu()
            mat_re[i_img][i_block] = np.array(trans(pred[0]))

    return mat_re

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='data/BenchmarkNoisyBlocksSrgb.mat', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the checkpoint to eval')
    parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')
    parser.add_argument('--model_type',default="DGF", help='type of model : DGF')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    mat_re = test(args)

    mat = scipy.io.loadmat(args.noise_dir)
    # print(mat['BenchmarkNoisyBlocksSrgb'].shape)
    del mat['BenchmarkNoisyBlocksSrgb']
    mat['DenoisedNoisyBlocksSrgb'] = mat_re
    # print(mat)
    scipy.io.savemat("SubmitSrgb.mat",mat)
