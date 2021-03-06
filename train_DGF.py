from model.MIRNet import MIRNet_DGF
import torch
import argparse
from torch.utils.data import DataLoader
from utils import losses
import os

# import h5py
from data.data_provider import SingleLoader_DGF,SingleLoader_DGF_raw,SingleLoader_filter
import torch.optim as optim
import numpy as np
# import model
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
# from collections import OrderedDict
from utils import robust_loss
from model.MIRNet_noise import MIRNet_noise
def train(args):
    torch.set_num_threads(args.num_workers)
    torch.manual_seed(0)
    if args.data_type == 'rgb':
        data_set = SingleLoader_DGF(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=args.burst_length)
    elif args.data_type == 'raw':
        data_set = SingleLoader_DGF_raw(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=args.burst_length)
    elif args.data_type == 'filter':
        data_set = SingleLoader_filter(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=args.burst_length)
    else:
        print("Data type not valid")
        exit()
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loss_func = losses.CharbonnierLoss().to(device)
    # loss_func_i = losses.LossAnneal_i()
    # loss_func = losses.AlginLoss().to(device)
    loss_func = losses.BasicLoss()
    adaptive = robust_loss.adaptive.AdaptiveLossFunction(
        num_dims=3*args.image_size**2, float_dtype=np.float32, device=device)
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if  args.model_type == "DGF":
        model = MIRNet_DGF(n_colors=args.n_colors,out_channels=args.out_channels,burst_length=args.burst_length).to(device)
    elif  args.model_type == "noise":
        model = MIRNet_noise(n_colors=args.n_colors,out_channels=args.out_channels,burst_length=args.burst_length).to(device)
    else:
        print(" Model type not valid")
        return
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    optimizer.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 10, 12, 14, 16], 0.8)
    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "model."+ k  # remove `module.`
            #     new_state_dict[name] = v
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    eps = 1e-4
    for epoch in range(start_epoch, args.epoch):
        for step, (image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr) in enumerate(data_loader):
            burst_noise = image_noise_lr.to(device)
            gt = image_gt_hr.to(device)
            image_gt_lr = image_gt_lr.to(device)
            image_noise_hr = image_noise_hr.to(device)
            pred_i, pred = model(burst_noise,image_noise_hr)
            # print(pred.size())
            loss_basic,_,_ = loss_func(pred,pred_i, gt,global_step)
            # loss_i = loss_func_i(10, pred_i, image_gt_lr)
            loss = loss_basic
            # bs = gt.size()[0]
            # diff = noise - gt
            # loss = torch.sqrt((diff * diff) + (eps * eps))
            # loss = loss.view(bs,-1)
            # loss = adaptive.lossfun(loss)
            # loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            if global_step % args.save_every == 0:
                print(len(average_loss._cache))
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t| loss_basic: {:.4f}\t|'
                      ' loss: {:.4f}\t| PSNR: {:.2f}dB\t.'
                      .format(global_step, epoch, step, loss_basic, loss, calculate_psnr(pred, gt)))
                # print(global_step, "PSNR  : ", calculate_psnr(pred, gt))
                print(average_loss.get_value())
            global_step += 1
        print('Epoch {} is finished.'.format(epoch))
        scheduler.step()


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir', '-n', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g', default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz', default=128, type=int, help='size of image')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='batch size')
    parser.add_argument('--burst_length', '-b', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', '-e', default=1000, type=int, help='batch size')
    parser.add_argument('--save_every', '-se', default=2, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le', default=1, type=int, help='loss_every')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--restart', '-r', action='store_true',
                        help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--model_type','-m' ,default="noise", help='type of model : DGF, noise')
    parser.add_argument('--data_type', '-d', default="rgb", help='type of model : rgb, raw')
    parser.add_argument('--n_colors', '-nc', default=3,type=int, help='number of color dim')
    parser.add_argument('--out_channels', '-oc', default=3,type=int, help='number of out_channels')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoints',
                        help='the checkpoint to eval')

    args = parser.parse_args()
    #
    train(args)