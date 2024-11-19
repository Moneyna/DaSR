from basicsr.data.transforms import augment
from basicsr.utils import img2tensor,tensor2img, imwrite
from basicsr.data.bsrgan_util import *

import cv2
import random
import argparse

import glob
import os

def create_lq(gt_path,scale=4,use_flip=False,use_rot=False):

    img_name = os.path.basename(gt_path)

    img_gt = cv2.imread(gt_path).astype(np.float32) / 255.

    img_gt = img_gt[:, :, [2, 1, 0]] # BGR to RGB

    img_lq, img_gt = degradation_bsrgan_plus(img_gt, shuffle_prob=0.5, sf=scale, lq_patchsize_h=img_gt.shape[0]// scale,lq_patchsize_w=img_gt.shape[1]//scale, use_crop=False)

    img_gt, img_lq = augment([img_gt, img_lq], use_flip,use_rot)
    img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

    return {
        'lq': img_lq,
        'gt': img_gt,
        'img_name':img_name
    }

def add_blur(img, sf=4,prob=0.5,kmin=2,kmax=11):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2*sf
    if random.random() < prob:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        k = anisotropic_Gaussian(ksize=2*random.randint(kmin,kmax)+3, theta=random.random()*np.pi, l1=l1, l2=l2) # [7,25]
    else:
        k = fspecial('gaussian', 2*random.randint(kmin,kmax)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img

def add_JPEG_noise(img,qmin=35,qmax=95):
    quality_factor = random.randint(qmin, qmax)
    img = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(uint2single(img), cv2.COLOR_BGR2RGB)
    return img

def degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.5, use_sharp=True, lq_patchsize_h=64,lq_patchsize_w=64, isp_model=None, use_crop=True):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize_h*sf or w < lq_patchsize_w*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    if use_sharp:
        img = add_sharpening(img)
    hq = img.copy()

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=sf)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=75, noise_level2=100)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img,qmin=70,qmax=95)
        elif i == 7:
            img = add_blur(img, sf=sf)
        elif i == 8:
            img = add_resize(img, sf=sf)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=75, noise_level2=100)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        else:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img,qmin=70,qmax=95)

    # random crop
    if use_crop:
        img, hq = random_crop(img, hq, sf, lq_patchsize_h,lq_patchsize_w)

    return img, hq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../data/', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default='../data/extra', help='Output folder')
    parser.add_argument('-s', '--scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--datasets',nargs='+',default=['Set5','Set14','B100','Urban100','DIV2K_VAL','Manga109'])
    parser.add_argument('--plus', action='store_true', help="save results or not")
    parser.add_argument('--use_flip', action='store_true', help="save results or not")
    parser.add_argument('--use_rot', action='store_true', help="save results or not")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output, exist_ok=True)

    sd = 'bsrgan_plus'

    for idx,dataset in enumerate(args.datasets):

        gt_base_path=os.path.join(args.input,dataset,'HR')
        if dataset=='DIV2K_VAL':
            gt_base_path = os.path.join(args.input, 'DIV2K', 'HR_VAL')

        gt_paths = sorted(glob.glob(os.path.join(gt_base_path, '*.png')))

        os.makedirs(os.path.join(args.output,sd,dataset,'HR'), exist_ok=True)
        os.makedirs(os.path.join(args.output, sd, dataset, 'LR','x'+str(args.scale)), exist_ok=True)

        for idx, path in enumerate(gt_paths):
            t_data=create_lq(path,scale=args.scale,plus=args.plus,use_flip=args.use_flip,use_rot=args.use_rot)
            img_name=t_data['img_name']
            (name,suffix)=os.path.splitext(img_name)

            hq_path=os.path.join(args.output,sd,dataset,'HR',f'{img_name}')
            lq_path = os.path.join(args.output,sd, dataset, 'LR','x'+str(args.scale),name+"_x"+str(args.scale)+suffix)
            hq=tensor2img(t_data['gt'])
            lq=tensor2img(t_data['lq'])

            print(hq_path)
            print(lq_path)
            # # save hq
            # imwrite(hq, hq_path)
            # # save lq
            imwrite(lq, lq_path)


if __name__ == '__main__':
    main()