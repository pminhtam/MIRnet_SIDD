import torch
import argparse
from model.MIRNet import MIRNet,MIRNet_kpn
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

# from torchsummary import summary

torch.set_num_threads(4)
torch.manual_seed(0)
torch.manual_seed(0)

def test(args):
    if args.model_type == "MIR":
        model = MIRNet()
    elif args.model_type == "KPN":
        model = MIRNet_kpn()
    else:
        print(" Model type not valid")
        return
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
    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['siddplus_valid_noisy_srgb']
    all_clean_imgs = scipy.io.loadmat(args.gt)['siddplus_valid_gt_srgb']
    # noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    # clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    i_imgs, _,_,_ = all_noisy_imgs.shape
    psnrs = []
    ssims = []
    # print(noisy_path)
    for i_img in range(i_imgs):
        noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img])).unsqueeze(0)
        noise = noise.to(device)
        begin = time.time()
        pred = model(noise)
        pred = pred.detach().cpu()
        gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img])))
        gt = gt.unsqueeze(0)
        psnr_t = calculate_psnr(pred, gt)
        ssim_t = calculate_ssim(pred, gt)
        psnrs.append(psnr_t)
        ssims.append(ssim_t)
        print(i_img, "   UP   :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
        if args.save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise KPN DGF " + args.model_type, fontsize=25)
            image_name = str(i_img) + "_"
            plt.axis("off")
            plt.suptitle(image_name + "   UP   :  PSNR : " + str(psnr_t) + " :  SSIM : " + str(ssim_t), fontsize=25)
            plt.savefig(os.path.join(args.save_img, image_name + "_" + args.checkpoint + '.png'), pad_inches=0)
    print("   AVG   :  PSNR : "+ str(np.mean(psnrs))+" :  SSIM : "+ str(np.mean(ssims)))


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='data/ValidationNoisyBlocksSrgb.mat', help='path to noise image file')
    parser.add_argument('--gt','-g', default='data/ValidationGtBlocksSrgb.mat', help='path to noise image file')
    # parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the checkpoint to eval')
    parser.add_argument('--image_size', '-sz', default=64, type=int, help='size of image')
    parser.add_argument('--model_type','-m' ,default="KPN", help='type of model : KPN, MIR')
    parser.add_argument('--save_img', "-s" ,default="", type=str, help='save image in eval_img folder ')

    args = parser.parse_args()
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    test(args)

