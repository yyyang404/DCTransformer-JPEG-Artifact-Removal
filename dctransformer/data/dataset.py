import os
import cv2
import random
import numpy as np
import albumentations as album

import torch
from torch import Tensor
import torch.nn.functional as F

from .image_ops import pad_to, rgb2ycbcr, unpad
from third_party.torchjpeg_tools import fdct, read_coefficients, read_coefficients_from_encoded_bytes, to_ycbcr


# data agumentation methods
def rand_aug(image, crop_size=(256, 256), p_crop = 1, p_channel_shuffle = 0.05, p_flip = 0.5, p_rotate = 0.5):
    train_transform = album.Compose([
        album.RandomCrop(crop_size[0], crop_size[1], p=p_crop),
        album.ChannelShuffle(p=p_channel_shuffle),
        album.OneOf([
            album.HorizontalFlip(p=p_flip),
            album.VerticalFlip(p=p_flip)
        ], p=1),
        album.RandomRotate90(p=p_rotate)
    ])
    return train_transform(image=image)


def _sample_qf(qf_range):
    qf_min, qf_max, qf_step = [int(x) for x in qf_range]
    return random.randrange(qf_min, qf_max + 1, qf_step)


def _read_coefficients_from_encoded_image(
    image: np.ndarray,
    qf: int,
):
    ok, encimg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    if not ok:
        raise RuntimeError("Failed to encode jpeg in memory.")
    return read_coefficients_from_encoded_bytes(encimg)


# speedy dataset: generating one batch from one HQ image
class SpeedyDCTCollocatedMapDataset(torch.utils.data.Dataset):

    def __init__(self,
                 in_paths, color_scale='gray', cbcr_interp=None,
                 batch_size=16, crop_size=(256, 256), is_check_size=False,
                 qf_range=(10, 90, 10)):  # qf_range: randint(min, max, step)

        self.in_paths = in_paths
        assert color_scale in ['color', 'gray']
        self.color_scale = color_scale
        self.interp_mode = cbcr_interp
        if self.interp_mode is None:
            self.cbcr_dsample_ratio = 2
        else:
            self.cbcr_dsample_ratio = 1

        self.batch_size = batch_size
        self.crop_size = crop_size
        self.qf_range = qf_range

        if is_check_size:
            print(f"number of input images: {len(self.in_paths)}")
            filtered_paths = []
            for path in self.in_paths:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1]:
                    filtered_paths.append(path)
            self.in_paths = filtered_paths
            print(f"number of filtered images: {len(filtered_paths)}")

    def __getitem__(self, index):

        if self.color_scale == 'color':
            x_y_batch = Tensor(self.batch_size, 64, self.crop_size[0] // 8, self.crop_size[1] // 8)
            x_cb_batch = Tensor(self.batch_size, 64, self.crop_size[0] // (8 * self.cbcr_dsample_ratio),
                                self.crop_size[1] // (8 * self.cbcr_dsample_ratio))
            x_cr_batch = Tensor(self.batch_size, 64, self.crop_size[0] // (8 * self.cbcr_dsample_ratio),
                                self.crop_size[1] // (8 * self.cbcr_dsample_ratio))
            qt_mat_batch = Tensor(self.batch_size, 3, 8, 8)
            y_batch = Tensor(self.batch_size, 192, self.crop_size[0] // 8, self.crop_size[1] // 8)

            img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)

            for in_batch_idx in range(self.batch_size):
                img_bgr_auged = rand_aug(img_bgr, crop_size=self.crop_size)['image']

                ############### preparing x
                
                rand_qf = _sample_qf(self.qf_range)


                dim, quant_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
                    img_bgr_auged,
                    rand_qf,
                )
                quant_mat = quant_mat.float()
                _, h, w, _, _ = y_coef.shape
                _, h_cbcr, w_cbcr, _, _ = cbcr_coef.shape
                cb_coef = cbcr_coef[0, ...]  # [16, 16, 8, 8]
                cr_coef = cbcr_coef[1, ...]  # [16, 16, 8, 8]
                x_y = y_coef.squeeze(0).float().multiply(
                    quant_mat[0, ...]).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]
                x_cb = cb_coef.float().multiply(
                    quant_mat[1, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]
                x_cr = cr_coef.float().multiply(
                    quant_mat[2, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]

                if self.interp_mode is not None:
                    x_cb = F.interpolate(  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2
                        x_cb[None, ...],
                        size=[h, w],
                        mode=self.interp_mode)[0, ...]  
                    x_cr = F.interpolate(  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2
                        x_cr[None, ...],
                        size=[h, w],
                        mode=self.interp_mode)[0, ...]  

                qt_mat_batch[in_batch_idx, ...] = quant_mat
                x_y_batch[in_batch_idx, ...] = x_y
                x_cb_batch[in_batch_idx, ...] = x_cb
                x_cr_batch[in_batch_idx, ...] = x_cr

                ############### preparing y
                img_rgb_auged = cv2.cvtColor(img_bgr_auged, cv2.COLOR_BGR2RGB).astype('float')
                img_rgb_tensor = Tensor(img_rgb_auged).permute(2, 0, 1)
                # convert ndarray (H x W x C) in [0, 255] to FloatTensor (C x H x W) in [0.0, 1.0]

                img_tensor_ycbcr = to_ycbcr(img_rgb_tensor, data_range=1.0)
                img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

                img_dct = fdct(img_tensor_ycbcr)
                img_dct_fold = img_dct.unfold(1, 8, 8).unfold(2, 8, 8).reshape(img_dct.shape[0], h, w, -1)
                y = img_dct_fold.permute(0, 3, 1, 2).reshape(-1, h, w)

                y_batch[in_batch_idx, ...] = y

            return qt_mat_batch, [x_y_batch, x_cb_batch, x_cr_batch], y_batch

        elif self.color_scale == 'gray':
            x_y_batch = Tensor(self.batch_size, 64, self.crop_size[0] // 8, self.crop_size[1] // 8)
            qt_mat_batch = Tensor(self.batch_size, 1, 8, 8)
            y_batch = Tensor(self.batch_size, 64, self.crop_size[0] // 8, self.crop_size[1] // 8)

            img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)

            for in_batch_idx in range(self.batch_size):
                img_bgr_auged = rand_aug(img_bgr, crop_size=self.crop_size)['image']

                ############### preparing x
                rand_qf = _sample_qf(self.qf_range)

                dim, quant_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
                    img_bgr_auged,
                    rand_qf,
                )
                quant_mat = quant_mat[0, ...].float()
                _, h, w, _, _ = y_coef.shape
                x_y = y_coef.squeeze(0).float().multiply(quant_mat).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]

                qt_mat_batch[in_batch_idx, ...] = quant_mat
                x_y_batch[in_batch_idx, ...] = x_y

                ############### preparing y
                img_rgb_auged = cv2.cvtColor(img_bgr_auged, cv2.COLOR_BGR2RGB).astype('float')
                img_rgb_tensor = Tensor(img_rgb_auged).permute(2, 0, 1)  # convert ndarray (H x W x C) in [0, 255] to FloatTensor (C x H x W) in [0.0, 1.0]

                img_tensor_ycbcr = to_ycbcr(img_rgb_tensor, data_range=1.0)
                img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

                img_dct = fdct(img_tensor_ycbcr[:1, ...])
                y = img_dct.squeeze(0).unfold(0, 8, 8).unfold(1, 8, 8).reshape(h, w, -1).permute(2, 0, 1)

                y_batch[in_batch_idx, ...] = y

            return qt_mat_batch, x_y_batch, y_batch

    def __len__(self):
        # return length of
        return len(self.in_paths)


# custom qf
def sample_custom_qf():
    if random.random() < 0.90: 
        return random.choice([10, 20, 30, 40, 50, 60])
    else:  # 10% chance
        return random.choice([70, 80, 90])  # Example range

def sample_custom_double_qf():
    r = random.random()
    if r < 0.80: 
        return random.choice(np.arange(10, 101, 5)), random.choice(np.arange(10, 101, 5))
    elif r < 0.85:  # 5% chance
        return 90, 10
    elif r < 0.90:  # 5% chance
        return 10, 90
    elif r < 0.95:  # 5% chance
        return 95, 75
    else:  # 10% chance
        return 75, 95

def sample_custom_shift():
    r = random.random()
    if r < 0.75:
        return random.choice(np.arange(0, 8, 1)), random.choice(np.arange(0, 8, 1))
    elif r < 0.875:  
        return 0, 0
    else:
        return 4, 4

# generalize dataset: generating one batch from batch_size HQ images
class DCTCollocatedMapDataset(torch.utils.data.Dataset):

    def __init__(self,
                 in_paths, color_scale='gray', cbcr_interp=None,
                 crop_size=(256, 256), is_check_size=False,
                 qf_range=(10, 90, 10), # qf_range: randint(min, max, step)
                 custom_qf=True,  
                 batch_size=16):  # fake, useless augument

        self.in_paths = in_paths
        assert color_scale in ['color', 'gray']
        self.color_scale = color_scale
        self.interp_mode = cbcr_interp
        if self.interp_mode is None:
            self.cbcr_dsample_ratio = 2
        else:
            self.cbcr_dsample_ratio = 1

        self.crop_size = crop_size
        self.qf_range = qf_range
        self.custom_qf = custom_qf
        if self.custom_qf:
            print("!!WARNING: custom qf is used. qf_range is ignored!!")

        if is_check_size:
            print(f"number of input images: {len(self.in_paths)}")
            filtered_paths = []
            for path in self.in_paths:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1]:
                    filtered_paths.append(path)
            self.in_paths = filtered_paths
            print(f"number of filtered images: {len(filtered_paths)}")

    def __getitem__(self, index):

        if self.color_scale == 'color':

            img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)
            img_bgr_auged = rand_aug(img_bgr, crop_size=self.crop_size)['image']

            ############### preparing x
            if self.custom_qf:
                rand_qf = sample_custom_qf()
            else:
                rand_qf = _sample_qf(self.qf_range)

            dim, qt_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
                img_bgr_auged,
                rand_qf,
            )
            qt_mat = qt_mat.float()
            _, h, w, _, _ = y_coef.shape
            _, h_cbcr, w_cbcr, _, _ = cbcr_coef.shape
            cb_coef = cbcr_coef[0, ...]  # [16, 16, 8, 8]
            cr_coef = cbcr_coef[1, ...]  # [16, 16, 8, 8]
            x_y = y_coef.squeeze(0).float().multiply(qt_mat[0, ...]).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]
            x_cb = cb_coef.float().multiply(qt_mat[1, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]
            x_cr = cr_coef.float().multiply(qt_mat[2, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]

            if self.interp_mode is not None:
                x_cb = F.interpolate(x_cb[None, ...], size=[h, w], mode=self.interp_mode)[
                    0, ...]  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2
                x_cr = F.interpolate(x_cr[None, ...], size=[h, w], mode=self.interp_mode)[
                    0, ...]  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2

            ############### preparing y
            img_rgb_auged = cv2.cvtColor(img_bgr_auged, cv2.COLOR_BGR2RGB).astype('float')
            img_rgb_tensor = Tensor(img_rgb_auged).permute(2, 0,
                                                           1)  # convert ndarray (H x W x C) in [0, 255] to FloatTensor (C x H x W) in [0.0, 1.0]

            img_tensor_ycbcr = to_ycbcr(img_rgb_tensor, data_range=1.0)
            img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

            img_dct = fdct(img_tensor_ycbcr)
            img_dct_fold = img_dct.unfold(1, 8, 8).unfold(2, 8, 8).reshape(img_dct.shape[0], h, w, -1)
            y = img_dct_fold.permute(0, 3, 1, 2).reshape(-1, h, w)

            return qt_mat, [x_y, x_cb, x_cr], y

        elif self.color_scale == 'gray':

            img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)

            if len(img_bgr.shape) == 2:
                img_bgr = np.expand_dims(img_bgr, axis=2)

            img_bgr_auged = rand_aug(img_bgr, crop_size=self.crop_size)['image']

            ############### preparing x
            if self.custom_qf:
                rand_qf = sample_custom_qf()
            else:
                rand_qf = _sample_qf(self.qf_range)

            dim, qt_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
                img_bgr_auged,
                rand_qf,
            )
            qt_mat = qt_mat[0, ...].float()
            _, h, w, _, _ = y_coef.shape
            x_y = y_coef.squeeze(0).float().multiply(qt_mat).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]

            ############### preparing y
            img_rgb_auged = cv2.cvtColor(img_bgr_auged, cv2.COLOR_BGR2RGB).astype('float')
            img_rgb_tensor = Tensor(img_rgb_auged).permute(2, 0,
                                                           1)  # convert ndarray (H x W x C) in [0, 255] to FloatTensor (C x H x W) in [0.0, 1.0]

            img_tensor_ycbcr = to_ycbcr(img_rgb_tensor, data_range=1.0)
            img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

            img_dct = fdct(img_tensor_ycbcr[:1, ...])
            y = img_dct.squeeze(0).unfold(0, 8, 8).unfold(1, 8, 8).reshape(h, w, -1).permute(2, 0, 1)

            return qt_mat, x_y, y

    def __len__(self):
        # return length of
        return len(self.in_paths)



class DCTCollocatedMapDataset_DoubleJPEG(torch.utils.data.Dataset):

    def __init__(self,
                 in_paths, color_scale='color', cbcr_interp=None,
                 crop_size=(256, 256), is_check_size=False,
                 qf_range=(10, 90, 10), # qf_range: randint(min, max, step)
                 custom_qf=True,  
                 custom_shift=True,
                 max_shift_h=7,
                 max_shift_w=7,
                 batch_size=16):  # fake, useless augument

        self.in_paths = in_paths
        assert color_scale in ['color', 'gray']
        self.color_scale = color_scale
        self.interp_mode = cbcr_interp
        if self.interp_mode is None:
            self.cbcr_dsample_ratio = 2
        else:
            self.cbcr_dsample_ratio = 1
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w

        self.crop_size = crop_size

        self.qf_range = qf_range
        self.custom_qf = custom_qf
        if self.custom_qf:
            print("!!WARNING: custom qf is used. qf_range is ignored!!")
        self.custom_shift = custom_shift
        if self.custom_shift:
            print("!!WARNING: custom shift is used. max_shift_h and max_shift_w are ignored!!")

        if is_check_size:
            print(f"number of input images: {len(self.in_paths)}")
            filtered_paths = []
            for path in self.in_paths:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1]:
                    filtered_paths.append(path)
            self.in_paths = filtered_paths
            print(f"number of filtered images: {len(filtered_paths)}")

    def __getitem__(self, index):

        if self.color_scale == 'color':

            img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)
            img_bgr_auged = rand_aug(
                img_bgr,
                crop_size=(self.crop_size[0] + self.max_shift_h, self.crop_size[1] + self.max_shift_w),
            )["image"]
            # here, the image is 8 pixel larger than the patch size.

            ############### preparing x
            if self.custom_qf:
                rand_qf1, rand_qf2 = sample_custom_double_qf()
            else:
                rand_qf1 = _sample_qf(self.qf_range)
                rand_qf2 = _sample_qf(self.qf_range)

            if self.custom_shift:
                shift_h, shift_w = sample_custom_shift()
            else:
                shift_h = random.randint(0, self.max_shift_h)
                shift_w = random.randint(0, self.max_shift_w)

            _, encimg_1 = cv2.imencode('.jpg', img_bgr_auged, [int(cv2.IMWRITE_JPEG_QUALITY), rand_qf1])
            img_bgr_auged_1 = cv2.imdecode(encimg_1, 3)

            # shifting training data
            img_bgr_auged = img_bgr_auged[shift_h:shift_h+self.crop_size[0], shift_w:shift_w+self.crop_size[1]]
            img_bgr_auged_1 = img_bgr_auged_1[shift_h:shift_h+self.crop_size[0], shift_w:shift_w+self.crop_size[1]]

            dim, qt_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
                img_bgr_auged_1,
                rand_qf2,
            )
            qt_mat = qt_mat.float()
            _, h, w, _, _ = y_coef.shape
            _, h_cbcr, w_cbcr, _, _ = cbcr_coef.shape
            cb_coef = cbcr_coef[0, ...]  # [16, 16, 8, 8]
            cr_coef = cbcr_coef[1, ...]  # [16, 16, 8, 8]
            x_y = y_coef.squeeze(0).float().multiply(qt_mat[0, ...]).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]
            x_cb = cb_coef.float().multiply(qt_mat[1, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]
            x_cr = cr_coef.float().multiply(qt_mat[2, ...]).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]

            if self.interp_mode is not None:
                x_cb = F.interpolate(x_cb[None, ...], size=[h, w], mode=self.interp_mode)[
                    0, ...]  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2
                x_cr = F.interpolate(x_cr[None, ...], size=[h, w], mode=self.interp_mode)[
                    0, ...]  # size=[x_y.shape[2],x_y.shape[3]]  or  scale_factor=2

            ############### preparing y
            img_rgb_auged = cv2.cvtColor(img_bgr_auged, cv2.COLOR_BGR2RGB).astype('float')
            img_rgb_tensor = Tensor(img_rgb_auged).permute(2, 0, 1)  # convert ndarray (H x W x C) in [0, 255] to FloatTensor (C x H x W) in [0.0, 1.0]

            img_tensor_ycbcr = to_ycbcr(img_rgb_tensor, data_range=1.0)
            img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

            img_dct = fdct(img_tensor_ycbcr)
            img_dct_fold = img_dct.unfold(1, 8, 8).unfold(2, 8, 8).reshape(img_dct.shape[0], h, w, -1)
            y = img_dct_fold.permute(0, 3, 1, 2).reshape(-1, h, w)

            return qt_mat, [x_y, x_cb, x_cr], y


    def __len__(self):
        # return length of
        return len(self.in_paths)


class ColorFolderEvaluationDataset(torch.utils.data.Dataset):

    def __init__(self, jpeg_paths, png_paths, color_interp_mode='bilinear'):

        self.in_paths = jpeg_paths
        self.gt_paths = png_paths
        assert len(self.in_paths) == len(self.gt_paths)
        self.channel = 3
        self.interp_mode = color_interp_mode

    def __getitem__(self, index):  # read images and masks

        filename = os.path.basename(self.gt_paths[index]).split('.')[0]
        img_gt_rgb = cv2.cvtColor(cv2.imread(self.gt_paths[index], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img_tensor_gt = Tensor(img_gt_rgb).permute(2, 0, 1)  / 255.

        dim, qt_mat, y_coef, cbcr_coef = read_coefficients(self.in_paths[index])
        
        ############### preparing x (NEW)

        _, h, w, _, _ = y_coef.shape
        x_y = y_coef.squeeze(0).float().multiply(qt_mat[0, ...].float()).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]

        _, h_cbcr, w_cbcr, _, _ = cbcr_coef.shape
        x_cb = cbcr_coef[0, ...]  # [16, 16, 8, 8]
        x_cr = cbcr_coef[1, ...]  # [16, 16, 8, 8]
        x_cb = x_cb.float().multiply(qt_mat[1, ...].float()).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]
        x_cr = x_cr.float().multiply(qt_mat[2, ...].float()).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]

        # print(f"img_tensor_gt.shape: {img_tensor_gt.shape}", f"x_y.shape: {x_y.shape}", f"x_cb.shape: {x_cb.shape}", f"x_cr.shape: {x_cr.shape}")
        return qt_mat, [x_y, x_cb, x_cr], img_tensor_gt, filename


    def __len__(self):
        # return length of
        return len(self.in_paths)



class EvaluationDataset(torch.utils.data.Dataset):

    def __init__(self, in_paths, color_scale='color', color_interp_mode='bilinear',
                 qf=10, color2gray=True):

        self.in_paths = in_paths
        self.img_names = [os.path.splitext(p.split("/")[-1])[0] for p in in_paths]
        assert color_scale in ['color', 'gray']
        self.color_scale = color_scale
        if self.color_scale == 'color':
            self.channel = 3
        else:
            self.channel = 1
        self.interp_mode = color_interp_mode
        self.qf = qf
        self.color2gray = color2gray

    def __getitem__(self, index):  # read images and masks

        img_bgr = cv2.imread(self.in_paths[index], cv2.IMREAD_UNCHANGED)
        img_bgr_mul16, pad = pad_to(img_bgr, 16)
        img_rgb_mul16 = cv2.cvtColor(img_bgr_mul16, cv2.COLOR_BGR2RGB)

        if self.color_scale == 'gray' and self.color2gray is True:
            img_y = rgb2ycbcr(img_rgb_mul16, only_y=False)[..., 0]
            img_to_encode = img_y
        elif self.color_scale == 'gray' and self.color2gray is False:
            img_to_encode = img_rgb_mul16[..., 0]
        else:
            img_to_encode = img_bgr_mul16

        dim, qt_mat, y_coef, cbcr_coef = _read_coefficients_from_encoded_image(
            img_to_encode,
            self.qf,
        )

        ############### preparing x (NEW)

        img_tensor_gt = Tensor(img_rgb_mul16).permute(2, 0, 1)

        _, h, w, _, _ = y_coef.shape
        x_y = y_coef.squeeze(0).float().multiply(qt_mat[0, ...].float()).view(h, w, -1).permute(2, 0, 1)  # [64, 32, 32]

        if self.color_scale == 'color':
            _, h_cbcr, w_cbcr, _, _ = cbcr_coef.shape
            x_cb = cbcr_coef[0, ...]  # [16, 16, 8, 8]
            x_cr = cbcr_coef[1, ...]  # [16, 16, 8, 8]
            x_cb = x_cb.float().multiply(qt_mat[1, ...].float()).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]
            x_cr = x_cr.float().multiply(qt_mat[2, ...].float()).view(h_cbcr, w_cbcr, -1).permute(2, 0, 1)  # [64, 32, 32]

        img_tensor_ycbcr = to_ycbcr(img_tensor_gt, data_range=1.0)
        img_tensor_ycbcr[0, ...] = img_tensor_ycbcr[0, ...] - 128  # centage

        if self.color_scale == 'color':
            img_dct = fdct(img_tensor_ycbcr)
            img_dct_fold = img_dct.unfold(1, 8, 8).unfold(2, 8, 8).reshape(img_dct.shape[0], h, w, -1)
            y = img_dct_fold.permute(0, 3, 1, 2).reshape(-1, h, w)
        else:
            img_dct = fdct(img_tensor_ycbcr[:1, ...])
            y = img_dct.squeeze(0).unfold(0, 8, 8).unfold(1, 8, 8).reshape(h, w, -1).permute(2, 0, 1)

        if self.color_scale == 'color':
            img_tensor_gt = unpad(img_tensor_gt, pad)
            return qt_mat, [x_y, x_cb, x_cr], y, img_tensor_gt / 255., pad
        elif self.color2gray is True:
            img_y = rgb2ycbcr(img_rgb_mul16, only_y=False)[..., 0]
            img_tensor_gt = Tensor(img_y)[None, ...]  # (RGB, H, W)
            img_tensor_gt = unpad(img_tensor_gt, pad)
            return qt_mat, x_y, y, img_tensor_gt / 255., pad
        else:  # self.color2gray is False:
            img_tensor_gt = img_rgb_mul16[None, ..., 0].astype(np.float32)
            img_tensor_gt = unpad(img_tensor_gt, pad)
            return qt_mat, x_y, y, img_tensor_gt / 255., pad

    def __len__(self):
        # return length of
        return len(self.in_paths)
