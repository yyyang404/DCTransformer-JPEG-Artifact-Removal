import torch
from torch import Tensor
import torch.nn as nn
from ..data.image_ops import reshape_image_from_frequencies, to_rgb, zigzag

# DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

'''
# --------------------------------------------
# Charbonnier Loss
# --------------------------------------------
'''


def get_outnorm(x: torch.Tensor, out_norm: str = '') -> float:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, out_norm: str = 'bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm


'''
# --------------------------------------------
# loss functions for GRAYSCALE training
# --------------------------------------------
'''


class GrayPixelDomainLoss(torch.nn.Module):
    def __init__(self, loss_type="l1", itrp_mode='nearest'):
        super(GrayPixelDomainLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == "smoothl1":
            self.pixel_loss = torch.nn.SmoothL1Loss()
        elif loss_type == "l1":
            self.pixel_loss = torch.nn.L1Loss()
        elif loss_type == "l2":
            self.pixel_loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"loss type [{loss_type}] is not found")
        self.itrp_mode = itrp_mode

    def forward(self, pred_y, target_y):
        # 将freq转换为pixel域 # -> (N, H, W)
        pred_img = reshape_image_from_frequencies(pred_y)
        target_img = reshape_image_from_frequencies(target_y)
        ls_pixel = self.pixel_loss(pred_img, target_img)
        return ls_pixel


class GrayDualDomainLoss(torch.nn.Module):
    def __init__(self, balance_ratio=255.):
        super(GrayDualDomainLoss, self).__init__()
        self.freq_loss = torch.nn.L1Loss()
        self.pixel_loss = CharbonnierLoss()
        self.balance_ratio = balance_ratio
        # frequency value range:(-1024, 1024), pixel value range:(0, 1)

    def forward(self, input, target):
        # 将freq转换为pixel域 # -> (N, H, W)
        input_pixel = reshape_image_from_frequencies(input)
        target_pixel = reshape_image_from_frequencies(target)

        ls_freq = self.freq_loss(input, target)
        ls_pixel = self.pixel_loss(input_pixel, target_pixel)

        # 计算综合的loss
        loss = self.balance_ratio * ls_pixel + ls_freq
        # print("loss = %.5f + %.5f = %.5f" % (self.balance_ratio*ls_pixel.item(), ls_freq.item(), loss))
        return loss


class GrayFreqEnhancedLoss(torch.nn.Module):
    def __init__(self, mode="low", device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(GrayFreqEnhancedLoss, self).__init__()
        self.freq_loss = torch.nn.SmoothL1Loss()

        enhance_mat = Tensor(8, 8).to(device)
        for val, pos in zigzag(8).items():
            enhance_mat[pos] = val / 64
        if mode == "low":
            enhance_mat = 1 - enhance_mat
        self.enhance_mat = enhance_mat.flatten().view(1, 64, 1, 1)

    def forward(self, input, target):
        enhanced_l = self.freq_loss(input.mul(self.enhance_mat), target.mul(self.enhance_mat))
        return enhanced_l


'''
# --------------------------------------------
# loss functions for COLORSCALE training
# --------------------------------------------
'''


class ColorFreqEnhancedLoss(torch.nn.Module):
    def __init__(self, mode="low", y_cbcr_ratio=(1.0, 0.5),
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(ColorFreqEnhancedLoss, self).__init__()
        self.freq_loss = torch.nn.SmoothL1Loss()
        self.y_cbcr_ratio = y_cbcr_ratio

        enhance_mat = Tensor(8, 8).to(device)
        for val, pos in zigzag(8).items():
            enhance_mat[pos] = val / 64
        if mode == "low":
            enhance_mat = 1 - enhance_mat
        self.enhance_mat_y = enhance_mat.flatten().view(1, 64, 1, 1) + 1
        self.enhance_mat_cbcr = torch.cat(
            [enhance_mat.flatten().view(1, 64, 1, 1) + 1, enhance_mat.flatten().view(1, 64, 1, 1) + 1], dim=1)

    def forward(self, input_y, input_cbcr, target_y, target_cbcr):
        enhanced_y_ls = self.freq_loss(input_y.mul(self.enhance_mat_y), target_y.mul(self.enhance_mat_y))
        enhanced_cbcr_ls = self.freq_loss(input_cbcr.mul(self.enhance_mat_cbcr), target_cbcr.mul(self.enhance_mat_cbcr))
        return enhanced_y_ls * self.y_cbcr_ratio[0] + enhanced_cbcr_ls * self.y_cbcr_ratio[1]


class ColorPixelDomainLoss(torch.nn.Module):
    def __init__(self, loss_type="l1", itrp_mode='nearest'):
        super(ColorPixelDomainLoss, self).__init__()
        loss_type = loss_type.lower()
        if loss_type == "smoothl1":
            self.pixel_loss = torch.nn.SmoothL1Loss()
        elif loss_type == "l1":
            self.pixel_loss = torch.nn.L1Loss()
        elif loss_type == "l2":
            self.pixel_loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f"loss type [{loss_type}] is not found")
        self.itrp_mode = itrp_mode

    def forward(self, pred_y, pred_cb, pred_cr, target_ycbcr):

        # 将freq转换为pixel域 # -> (N, H, W)
        pred_y_chn = reshape_image_from_frequencies(pred_y)
        pred_cb_chn = reshape_image_from_frequencies(pred_cb)
        pred_cr_chn = reshape_image_from_frequencies(pred_cr)
        pred_ycbcr_img = torch.cat([pred_y_chn, pred_cb_chn, pred_cr_chn], dim=1)

        pred_img = to_rgb(pred_ycbcr_img, data_range=1.0)
        target_img = to_rgb(reshape_image_from_frequencies(target_ycbcr), data_range=1.0)
        ls_pixel = self.pixel_loss(pred_img, target_img)
        return ls_pixel


class ColorDualDomainLoss(torch.nn.Module):
    def __init__(self, balance_ratio=255.):
        super(ColorDualDomainLoss, self).__init__()
        self.freq_loss = torch.nn.L1Loss()
        self.pixel_loss = CharbonnierLoss()
        self.balance_ratio = balance_ratio
        # frequency value range:(-1024, 1024), pixel value range:(0, 1)

    def forward(self, input_y, input_cb, input_cr, target_ycbcr):
        input_ycbcr = torch.cat([input_y, input_cb, input_cr], dim=1)
        input_rgb_img = to_rgb(reshape_image_from_frequencies(input_ycbcr), data_range=1.0)
        target_rgb_img = to_rgb(reshape_image_from_frequencies(target_ycbcr), data_range=1.0)

        ls_freq = self.freq_loss(input_ycbcr, target_ycbcr)
        ls_pixel = self.pixel_loss(input_rgb_img, target_rgb_img)

        # 计算综合的loss
        loss = self.balance_ratio * ls_pixel + ls_freq
        # print("loss = %.5f + %.5f = %.5f" % (self.balance_ratio*ls_pixel.item(), ls_freq.item(), loss))
        return loss


if __name__ == "__main__":
    pass
    # mats = FreqWiseEnhancedLoss(color_scale="color", mode="low", y_cb_cr_ratio=[2.0, 0.5, 0.5]).enhance_mat.view(3,8,8).cpu().numpy()
    # visualize_heatmaps(y=mats[0,...], cb=mats[1,...], cr=mats[2,...], figsize=(20,6), is_annot=True)
    #
    # ycbcr_loss = YCbCr_FreqWiseEnhancedLoss(mode="low", y_cbcr_ratio=[1.0, 0.5])
    # mat_y = ycbcr_loss.enhance_mat_y.view(1, 8, 8).cpu().numpy()
    # mat_cbcr = ycbcr_loss.enhance_mat_cbcr.view(2, 8, 8).cpu().numpy()
    # visualize_heatmaps(y=mat_y[0, ...] * ycbcr_loss.y_cbcr_ratio[0], cbcr=mat_cbcr[0, ...] * ycbcr_loss.y_cbcr_ratio[1], figsize=(10, 3), is_annot=True)
