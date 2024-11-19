import argparse
import cv2
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
from tqdm import tqdm
import torch
import pyiqa
import warnings

from utils import img2tensor, tensor2img, imwrite
from archs.dasr_arch import dasr

#pyiqa 0.1.10
val={'metrics':{'psnr':{'type':'psnr','crop_border':0,'test_y_channel':True},
                'ssim':{'type':'ssim','crop_border':0,'test_y_channel':True},
                'lpips':{'type':'lpips','better':'lower'},
                'topiq_fr': {'type': 'topiq_fr'},
                'qalign':{'type':'qalign'}
}}

os.environ['CUDA_LAUNCH_BLOCKING']="1"
warnings.filterwarnings('ignore')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-igt', '--gt', type=str, default=None, help='gt image or folder')
    parser.add_argument('-w', '--weight', type=str, default='../experiments/pretrained_models/dasr_model_g.pth', help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--save_imgs',action='store_true',help="save results or not")
    parser.add_argument('-params',nargs='+',help="model parameters")
    parser.add_argument('--save_metrics',action='store_true',help="save metrics results or not")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    weight_path = args.weight

    if args.save_metrics:
        title='img_name,'
    metric_funcs = {}
    for _, opt in val['metrics'].items():
        mopt = opt.copy()
        name = mopt.pop('type', None)
        if args.save_metrics:
            title+=name+','
        mopt.pop('better', None)
        metric_funcs[name] = pyiqa.create_metric(name, device='cuda', **mopt)

    args.params[0]=True if any(item in args.params[0] for item in ['True','true']) else False

    # set up the model
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

    # 单个文件输出
    if args.save_metrics:
        file_list=args.output.split('/')
        dataset=file_list[-1]
        fs='/'.join(file_list[:-1])
        os.makedirs(fs, exist_ok=True)
        sfile_path=os.path.join(fs,file_list[-2]+'_'+dataset+'.txt')
        sfile = open(sfile_path, 'w')
        sfile.write(title[:-1]+'\n')
        sfile.flush()

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*.png')))

    pbar = tqdm(total=len(paths), unit='image')
    metric_results = {
        metric: 0
        for metric in val['metrics'].keys()
    }
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        line = img_name+','
        # LQ
        if args.gt:
            gt_path = os.path.join(args.gt,img_name[:-7]+'.png') # xxxx_x4.png
        if args.params[0] == False:
            gt_path=path
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        if args.gt:
            gt=cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
            gt_tensor = img2tensor(gt).to(device) / 255.
            #gt_tensor = img2tensor(gt) / 255.
            gt_tensor = gt_tensor.unsqueeze(0)

        h, w = img_tensor.shape[2:]
        max_size=1000**2 #500**2 #800**2
        if h * w < max_size:
            output,_ = sr_model.test(img_tensor)
        else:
            output,_ = sr_model.test_tile(img_tensor,tile_size=400)

        if args.gt:
            H,W=gt_tensor.shape[2:]
            new_h=min(h*args.out_scale,H)
            new_w=min(w*args.out_scale,W)

            metric_data = [output[:,:,:new_h,:new_w].cpu(), gt_tensor[:,:,:new_h,:new_w].cpu()]
        for name, opt_ in val['metrics'].items():
            if args.gt ==None and not any(item in name for item in ['niqe','maniqa','topiq_nf','clipiqa','musiq','qalign']):
                continue
            if any(item in name for item in ['niqe','maniqa','topiq_nf','clipiqa','musiq','qalign']):
                tmp_result=metric_funcs[name](output)
            else:
                tmp_result = metric_funcs[name](*metric_data)
            metric_results[name] += tmp_result.item()
            if args.save_metrics:
                line+=str(tmp_result.item())+','
        if args.save_metrics:
            sfile.write(line[:-1]+'\n')
            sfile.flush()

        output_img = tensor2img(output)
        if args.save_imgs:
            save_path = os.path.join(args.output, f'{img_name}')
            imwrite(output_img, save_path)
            # LR save
            #imwrite(tensor2img(img_tensor),os.path.join(args.output,img_name[:-7]+'_LR.png'))
        pbar.update(1)
    pbar.close()
    for metric in metric_results.keys():
        metric_results[metric] /= (idx + 1)
    print(metric_results)

if __name__ == '__main__':
    main()
