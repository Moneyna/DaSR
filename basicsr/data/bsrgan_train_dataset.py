import numpy as np
from torch.utils import data as data

from data.bsrgan_util import degradation_bsrgan,degradation_bsrgan_plus
from data.transforms import augment
from utils import FileClient, img2tensor
from utils.registry import DATASET_REGISTRY

from .data_util import make_dataset, input_mask_with_noise

import cv2
import random


def random_resize(img, scale_factor_h=1.,scale_factor_w=1.0):
    return cv2.resize(img, None, fx=scale_factor_w, fy=scale_factor_h, interpolation=cv2.INTER_CUBIC)


def random_crop(img, out_size):
    h, w = img.shape[:2]
    # print("h-out_size=",h-out_size)
    # print("w-out_size=",w-out_size)
    rnd_h = random.randint(0, h - out_size)
    rnd_w = random.randint(0, w - out_size)
    return img[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]


@DATASET_REGISTRY.register()
class BSRGANTrainDataset(data.Dataset):
    """Synthesize LR-HR pairs online with BSRGAN for image restoration.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(BSRGANTrainDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
                
        self.gt_paths = make_dataset(self.gt_folder)

    def __getitem__(self, index):
        
        scale = self.opt['scale']

        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.

        img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB
        #gt_size = self.opt['gt_size']

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            input_gt_h, input_gt_w = img_gt.shape[0], img_gt.shape[1]
            if input_gt_h < gt_size or input_gt_w < gt_size:
                pad_h = max(0, gt_size - input_gt_h)
                pad_w = max(0, gt_size - input_gt_w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            if self.opt['use_resize_crop']:
                input_gt_size_h = img_gt.shape[0]
                input_gt_size_w = img_gt.shape[1]
                input_gt_random_size_h = random.randint(gt_size, input_gt_size_h)
                input_gt_random_size_w = random.randint(gt_size, input_gt_size_w)
                resize_factor_h = input_gt_random_size_h / input_gt_size_h
                resize_factor_w = input_gt_random_size_w / input_gt_size_w
                img_gt = random_resize(img_gt, resize_factor_h,resize_factor_w)

            img_gt = random_crop(img_gt, gt_size)

        img_lq, img_gt = degradation_bsrgan(img_gt, sf=scale, lq_patchsize_h=img_gt.shape[0] // scale, lq_patchsize_w=img_gt.shape[1] // scale, use_crop=False)
        if self.opt.get('if_mask'):
            img_lq = input_mask_with_noise(img_lq, mask1=self.opt['mask1'], mask2=self.opt['mask2'])
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
                                     self.opt['use_rot'])
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
