import torch
import argparse
from model.MIRNet import MIRNet,MIRNet_kpn

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

# from torchsummary import summary
def load_data(image_file):
    image_noise = transforms.ToTensor()(Image.open(image_file).convert('RGB'))
    image_noise = image_noise.unsqueeze(0)
    # _,c,h,w = image_noise.size()
    # hh = int(h/2)
    # ww = int(w/2)
    # image_noises = torch.cat([image_noise[:,:,:hh,:ww],image_noise[:,:,hh:,:ww],image_noise[:,:,:hh,ww:],image_noise[:,:,hh:,ww:]])
    # image_noise = image_noise.unsqueeze(0)
    return image_noise
def split_tensor(ten):
    _,c,h,w = ten.size()
    hh = int(h / 2)
    ww = int(w / 2)
    ten_cat = torch.cat([ten[:,:,:hh,:ww],ten[:,:,hh:,:ww],ten[:,:,:hh,ww:],ten[:,:,hh:,ww:]])
    return ten_cat
def merge_tensor(tens):
    # print(tens[0].size())
    c,hh,ww = tens[0].size()
    ten_full = torch.zeros((c,hh*2,ww*2))
    ten_full[ :, :hh, :ww] = tens[0]
    ten_full[ :, hh:, :ww] = tens[1]
    ten_full[ :, :hh, ww:] = tens[2]
    ten_full[ :, hh:,ww:] = tens[3]
    return ten_full
def test(args):
    model = MIRNet()

    # summary(model,[[3,128,128],[0]])
    # exit()
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    # except:
    #     print('=> no checkpoint file to be loaded.')    # model.load_state_dict(state_dict)
    #     exit(1)
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    # noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    test_img = glob.glob("/vinai/tampm2/cityscapes_noise/gtFine/val/*/*_gtFine_color.png")
    if not os.path.exists(args.save_img):
        os.makedirs(args.save_img)
    for i in range(len(test_img)):
        # print(noisy_path[i])
        img_path = os.path.join(args.noise_dir,test_img[i].split("/")[-1].replace("_gtFine_color","_leftImg8bit"))
        print(img_path)
        image_noise = load_data(img_path)
        image_noises1 = split_tensor(image_noise)
        preds1 = []
        for image1 in image_noises1:
            image1 = image1.unsqueeze(0)
            image_noises2 = split_tensor(image1)
            preds2 = []
            for image2 in image_noises2:
                image2 = image2.unsqueeze(0)
                image_noises3 = split_tensor(image2)
                preds3 = []
                for image3 in image_noises3:
                    image3 = image3.unsqueeze(0).to(device)
                    print(image3.size())
                    pred3 = model(image3)
                    pred3 = pred3.detach().cpu().squeeze(0)
                    preds3.append(pred3)

                pred2 = merge_tensor(preds3)
                preds2.append(pred2)
            pred1 = merge_tensor(preds2)
            preds1.append(pred1)
        pred = merge_tensor(preds1)
        pred = trans(pred)
        name_img = img_path.split("/")[-1].split(".")[0]
        pred.save(args.save_img + "/" + name_img+".png")
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/vinai/tampm2/cityscapes_denoising_dataset/cityscape/noise/', help='path to noise image file')
    # parser.add_argument('--gt','-g', default='data/ValidationGtBlocksSrgb.mat', help='path to noise image file')
    # parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='mir',
                        help='the checkpoint to eval')
    # parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')
    parser.add_argument('--model_type','-m' ,default="MIR", help='type of model : KPN, MIR')
    parser.add_argument('--save_img', "-s" ,default="img_city", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    test(args)

