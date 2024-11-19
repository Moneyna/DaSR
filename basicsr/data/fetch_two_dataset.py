import numpy as np
from torch.utils import data as data

from .bsrgan_util import degradation_bsrgan, degradation_bsrgan_plus
from .transforms import augment, paired_random_crop
from utils import FileClient, img2tensor
from utils.registry import DATASET_REGISTRY

from .data_util import make_dataset

import cv2
import random


def random_resize(img, scale_factor_h=1., scale_factor_w=1.0):
    return cv2.resize(img, None, fx=scale_factor_w, fy=scale_factor_h, interpolation=cv2.INTER_CUBIC)


def random_crop(img, out_size):
    h, w = img.shape[:2]
    rnd_h = random.randint(0, h - out_size)
    rnd_w = random.randint(0, w - out_size)
    return img[rnd_h: rnd_h + out_size, rnd_w: rnd_w + out_size]

class patch_input():
    def __init__(self,patch_size=8):
        self.patch_size=patch_size

    def __len__(self):
        return self.nH*self.nW

    def de_patchfy(self, x):  # 输入为B,H,W,C
        x = x.reshape( self.nH , self.nW, self.patch_size,self.patch_size,self.shape[-1]).transpose(0,2,1,3,4)
        x = x.reshape(self.shape)
        return x

    def patchfy(self, x):
        H, W, C = x.shape

        self.shape = x.shape
        self.nH = H // self.patch_size
        self.nW = W // self.patch_size

        x = x.reshape(self.nH, self.patch_size, self.nW, self.patch_size, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(self.nH*self.nW, self.patch_size, self.patch_size,C)
        # print("[patchfy]x.shape=",x.shape)
        return x

@DATASET_REGISTRY.register()
class FetchTwoDataset(data.Dataset):
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
        super(FetchTwoDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_paths = make_dataset(self.gt_folder)

        self.patch_size = opt['patch_size']
        scale = self.opt['scale']
        self.mode = self.opt['data_mode']


        if self.mode=='paired':
            self.lq_forder = opt['dataroot_lq']
            self.lq_paths = make_dataset(self.lq_forder)

        self.hq_img_patch = patch_input(self.patch_size)
        self.lq_img_patch = patch_input(self.patch_size//scale)


    def __getitem__(self, index):

        scale = self.opt['scale']

        gt_path = self.gt_paths[index]
        img_gt = cv2.imread(gt_path).astype(np.float32) / 255.

        img_gt = img_gt[:, :, [2, 1, 0]]  # BGR to RGB

        if self.mode == 'paired':
            lq_path = self.lq_paths[index]
            if gt_path != lq_path:
                t1 = gt_path.split("/")
                t2 = lq_path.split("/")
                t2[-1] = t1[-1][:-4] + t2[-1][-7:]
                lq_path = '/'.join(t2)
            img_lq = cv2.imread(lq_path).astype(np.float32) / 255.

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            if self.opt['use_resize_crop']:
                input_gt_size_h = img_gt.shape[0]
                input_gt_size_w = img_gt.shape[1]
                input_gt_random_size_h = random.randint(gt_size, input_gt_size_h)
                input_gt_random_size_w = random.randint(gt_size, input_gt_size_w)
                resize_factor_h = input_gt_random_size_h / input_gt_size_h
                resize_factor_w = input_gt_random_size_w / input_gt_size_w
                img_gt = random_resize(img_gt, resize_factor_h, resize_factor_w)

                if self.mode=='paired':
                    img_lq = random_resize(img_lq, scale_factor_h=resize_factor_h,scale_factor_w=resize_factor_w)

            if self.mode=='paired':
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            else:
                img_gt = random_crop(img_gt, gt_size)

        if self.mode !='paired':
            img_lq, img_gt = degradation_bsrgan(img_gt, sf=scale, lq_patchsize_h=img_gt.shape[0] // scale, lq_patchsize_w=img_gt.shape[1] // scale, use_crop=False)

        img_hq_list = self.hq_img_patch.patchfy(img_gt)
        img_lq_list = self.lq_img_patch.patchfy(img_lq)

        #TODO: 循环数据增广
        hq_list=[]
        lq_list=[]
        for i in range(len(self.lq_img_patch)):
            [img_gt_aug, img_lq_aug],status = augment([img_hq_list[i,...], img_lq_list[i,...]], self.opt['use_flip'], self.opt['use_rot'],return_status=True)
            while not (status[0] or status[1] or status[2]):
                [img_gt_aug, img_lq_aug], status = augment([img_hq_list[i,...], img_lq_list[i,...]], self.opt['use_flip'], self.opt['use_rot'],return_status=True)
            hq_list.append(img_gt_aug)
            lq_list.append(img_lq_aug)
        img_gt_aug=self.hq_img_patch.de_patchfy(np.array(hq_list))
        img_lq_aug=self.lq_img_patch.de_patchfy(np.array(lq_list))

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        img_gt_aug, img_lq_aug = img2tensor([img_gt_aug, img_lq_aug], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_aug': img_lq_aug,
            'gt_aug': img_gt_aug,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.gt_paths)
