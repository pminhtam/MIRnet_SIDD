import gc
import numpy as np
import h5py


def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    # print('img_shape: ' + str(img_shape))
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)

    del raw_im
    gc.collect()

    return out

def unpack_raw(raw_pack):
    """Un Packs Bayer image to 4 channels (h, w) <-- (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    img_shape = raw_pack.shape
    # print(raw_pack.shape)
    # print('img_shape: ' + str(img_shape))
    h = img_shape[0]
    w = img_shape[1]
    # print(h,w)
    out = np.zeros((h*2,w*2))
    # print("out.shape  : ",out.shape)
    # print("out[0:h:2, 0:w:2].shape  : ",out[0:h*2:2, 0:w*2:2].shape)
    # print("raw_pack.shape  : ",raw_pack.shape)
    # print(raw_pack[:, :,0].shape)
    out[0:h*2:2, 0:w*2:2] = raw_pack[:, :,0]
    out[0:h*2:2, 1:w*2:2] = raw_pack[:, :,1]
    out[1:h*2:2, 1:w*2:2] = raw_pack[:, :,2]
    out[1:h*2:2, 0:w*2:2] = raw_pack[:, :,3]

    gc.collect()

    return out

def read_raw(img_path):
    with h5py.File(img_path, 'r') as f:  # (use this for .mat files with -v7.3 format)
        # raw = f[list(f.keys())[0]]  # use the first and only key
        # print(list(f.keys()))
        raw = f['x']  # use the first and only key
        # input_image  = pack_raw(raw)
        # print(np.array(raw))
        # print(input_image.shape)
        # unpack_input_image  = unpack_raw(input_image)
        # print(input_image)
        # print(unpack_input_image)
        # input_image = np.transpose(raw)  # TODO: transpose?
        # input_image = np.expand_dims(pack_raw(raw), axis=0)
        # print(raw)
        # input_image = np.nan_to_num(raw)
        # input_image = np.clip(input_image, 0.0, 1.0)
        input_image = np.array(raw)
    return input_image

from glob import glob
if __name__ == "__main__":
    files_ = glob('/home/dell/Downloads/noise_raw/0001_NOISY_RAW/*')
    for fi in files_:
        input_image = read_raw(fi)
        # print(input_image)
        # print(wb)
        exit(0)