import argparse
import cv2
import glob
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from tqdm import tqdm
import torch
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt

from basicsr.archs.dasr_arch import dasr

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings('ignore')

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')

def plt_show(x,save_path=""):
    plt.figure()
    plt.imshow(x.cpu().numpy(), cmap='PuBu')
    plt.colorbar()
    if save_path!="":
        plt.savefig(save_path)

def img_softmax(x,save_path=""): # x.shape=[1,h,N,Cn]
    sft1 = nn.Softmax(dim=-1)

    a12 = sft1(x)
    a2 = a12.mean(1).mean(-2).reshape(16, 16)
    plt_show(a2,save_path+'img_softmax.png')
    return a2

def img_wo_softmax(x,nh,nw,save_path=""):
    min_value = x.min()
    max_value = x.max()

    normalized_tensor = (x - min_value) / (max_value - min_value)
    a2 = normalized_tensor.mean(1).mean(-1).reshape(nh, nw)
    plt_show(a2, save_path + 'img_wo_softmax.png')
    return a2


def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../data/bsrgan_plus/Set5/LR/x4', help='Input image or folder')
    parser.add_argument('-igt', '--gt', type=str, default='../data/bsrgan_plus/Set5/HR', help='gt image or folder')
    parser.add_argument('-w', '--weight', type=str, default='../experiments/DaSR_LQ_stage_256_256_wGAN/models/net_g_best_.pth',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='attn_map/Set5', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('-params', nargs='+',default=[True, 256, 256, 8, 3, 64, 128, 256, 6, 0, 256, 0, 4],
                        help="model parameters")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weight_path = args.weight

    args.params[0]=bool(args.params[0])
    sr_model = dasr(upscale=args.out_scale,
                    LQ_stage=args.params[0],
                    codebook_n=int(args.params[1]),
                    codebook_dim=int(args.params[2]),
                    num_heads=int(args.params[3]),
                    mode=int(args.params[4]),
                    channel_list=[int(args.params[5]),int(args.params[6]),int(args.params[7]),int(args.params[7])],
                    nTRG=int(args.params[8]),
                    nTAG=int(args.params[9]),
                    psN=int(args.params[10]),
                    nMHA=int(args.params[11]),
                    d_MHA=int(args.params[12])).to(device)

    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=True)
    sr_model.eval()

    print(args.output)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*.png')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        h, w = img_tensor.shape[2:]
        max_size = 1000 ** 2
        if h * w < max_size:
            output, attn_list = sr_model.test(img_tensor)
        else:
            output, attn_list = sr_model.test_tile(img_tensor, tile_size=400)

        # save attn_list
        if args.params[0]==True:
            save_path = os.path.join(args.output,f'{img_name[:-7]}')
        else:
            save_path = os.path.join(args.output,f'{img_name[:-4]}')

        for i,x in enumerate(attn_list):
            if args.params[0]==True:
                attn_map = x[:, :, :h//2, :w//2, :]
            else:
                attn_map = x[:, :, :h//8, :w//8, :]

            _, head, nh, nw, Cn = attn_map.shape

            attn_map = attn_map.reshape(1, head, -1, Cn)
            sp=os.path.join(save_path+"_A"+str(i)+"_")
            a1 = img_softmax(attn_map, sp)
            a2 = img_wo_softmax(attn_map, nh, nw, sp)

        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
