import h5py
import numpy as np
import glob
import os
import argparse
from utils.raw_util import pack_raw,read_raw
from utils.crop_image import crop_image
if __name__ == "__main__":
    # image_path = "/home/dell/Downloads/0001_GT_RAW/0001_GT_RAW_003.MAT"
    # image_path = "/home/dell/Downloads/0001_NOISY_RAW/0001_NOISY_RAW_001.MAT"
    # image_path = "/home/dell/Downloads/noise_raw/0001_NOISY_RAW/0001_NOISY_RAW_001.MAT"
    # raw_image = rawpy.imread(image_path).raw_image
    # f = h5py.File(image_path)
    # arrays_noise = {}
    # print(f.items())
    # for k, v in f.items():
    #     # print(k)
    #     arrays_noise[k] = np.array(v)
    #     # print(v.shape)
    # noisy_img = arrays_noise['x']
    # print(arrays_noise)
    # print(np.max(noisy_img))

    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--image_path', '-i', default='/home/dell/Downloads/noise_raw/0001_NOISY_RAW/', help='path to noise folder image')
    # parser.add_argument('--image_path', '-n', default='/home/dell/Downloads/noise_raw/split/', help='path to noise folder image')
    parser.add_argument('--save_path', '-s', default='/home/dell/Downloads/noise_raw/split/', help='path to gt folder image')
    parser.add_argument('--crop_size', '-c', default=256, type=int, help='Crop size')

    args = parser.parse_args()
    #
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    files_ = glob.glob(os.path.join(args.image_path,"*"))
    crop_size = (args.crop_size,args.crop_size)

    for fi in files_:
        name = fi.split('/')[-1].split(".")[0]
        input_image = read_raw(fi)
        print(input_image.shape)
        pack_img = pack_raw(input_image)
        print(pack_img.shape)
        list_img_crop = crop_image(pack_img, crop_size)
        # print(input_image)
        # f = h5py.File(os.path.join(args.save_path,name + "__"+".MAT"), "w")
        # f.create_dataset('y', data=input_image,dtype='float32')
        # f.close()
        for i in range(len(list_img_crop)):
            f = h5py.File(os.path.join(args.save_path,name + "_" +  str(i)  + ".MAT"), "w")
            f.create_dataset('x', data=list_img_crop[i],dtype='float32')
            f.close()

