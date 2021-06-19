import os
import time
import torch
import imageio
import numpy as np

# from convert.converter import torch2onnx, onnx2keras, keras2tflite
# from denoiser.networks.denoising_rgb import DenoiseNet
# from denoiser.utils import load_checkpoint
from model.MIRNet import MIRNet
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint


img_path = '/home/dell/Downloads/FullTest/noisy/2_1.png'
model_path = '../../denoiser/pretrained_models/denoising/sidd_rgb.pth'
output_path = 'models/denoiser_rgb.onnx'

input_node_names = ['input_image']
output_nodel_names = ['output_image']

# torch_model = DenoiseNet()
# load_checkpoint(torch_model, model_path, 'cpu')
checkpoint = load_checkpoint("../checkpoints/mir/", False, 'latest')
state_dict = checkpoint['state_dict']
torch_model = MIRNet()
print(torch_model)
exit(0)
torch_model.load_state_dict(state_dict)
torch_model.eval()
img = imageio.imread(img_path)
img = img[0:256,0:256,:]
print(img.shape)
img = np.asarray(img, dtype=np.float32) / 255.

img_tensor = torch.from_numpy(img)
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

print('Test forward pass')
s = time.time()
# with torch.no_grad():
#     enhanced_img_tensor = torch_model(img_tensor)
#     enhanced_img_tensor = torch.clamp(enhanced_img_tensor, 0, 1)
#     enhanced_img = (enhanced_img_tensor.permute(0, 2, 3, 1).squeeze(0)
#                     .cpu().detach().numpy())
#     enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype('uint8')
#     imageio.imwrite('../img/denoised.jpg', enhanced_img)
print('- Time: ', time.time() - s)
print(torch_model)
# torch.quantization.fuse_modules(torch_model, [['Conv2d', 'relu']], inplace=True)



import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(torch_model, {torch.nn.Linear,torch.nn.Conv2d}, dtype=torch.qint8)
# # set quantization config for server (x86)
# torch_model.qconfig = torch.quantization.default_qconfig('fbgemm')
# # insert observers
# torch.quantization.prepare(torch_model, inplace=True)
# # Calibrate the model and collect statistics
# # convert to quantized version
# torch.quantization.convert(torch_model, inplace=True)


torchscript_model = torch.jit.trace(quantized_model,img_tensor)
import torch.utils.mobile_optimizer as mobile_optimizer

torchscript_model_optimized = torch.utils.mobile_optimizer.optimize_for_mobile(torchscript_model)
torch.jit.save(torchscript_model_optimized, "model.pt")
