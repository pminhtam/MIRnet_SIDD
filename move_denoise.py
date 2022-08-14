import glob
from shutil import copy2
import os


# img_paths = glob.glob("img_city/*.png")
test_img = glob.glob("/vinai/tampm2/cityscapes_noise/leftImg8bit_ori/val/*/*.png")

# for img_path in img_paths:
#     copy2(img_path,"../cityscape/gt/")
save_path = "/vinai/tampm2/cityscapes_noise/leftImg8bit_denoise/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for img in test_img:
    img_name = img.split("/")[-1]
    copy2(os.path.join("img_city10",img_name),os.path.dirname(img.replace("/leftImg8bit/","/leftImg8bit_denoise/")))