import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, color, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, color, height, width)

            core_out[K] = torch.einsum('ijklnop,ijlmnop->ijkmnop', [t1, t2]).view(batch_size, N, K * K, 1, height, width)

            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, 1, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        # for key, data in core.items():
        #     print(key, data.size())
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if len(img_stack) == 0:
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[..., i:i + height, j:j + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1]**2 - kernel[index]**2) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias

        if color == 1:
            pred_img_i = pred_img_i.squeeze(2)

        try:
            while len(pred_img_i.size()) > len(white_level.size()):
                white_level = white_level.unsqueeze(-1)
            white_level = white_level.type_as(pred_img_i).expand_as(pred_img_i)
        except:
            pass

        pred_img_i = pred_img_i / white_level
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        # print('pred_img:', pred_img.size())
        return pred_img_i, pred_img