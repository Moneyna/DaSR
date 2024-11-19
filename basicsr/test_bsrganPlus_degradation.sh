#! /bin/bash
#e.g. ./basicsr/test_bsrganPlus_degradation.sh pretrain_model DaSR_pretrain true 256 256 8 3 64 128 256 0 6 0 256 0.0 0 4

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/DIV2K_VAL/LR/x4 -igt ../data/bsrgan/DIV2K_VAL/HR/ -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/Div2Kval --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/Set5/LR/x4 -igt ../data/bsrgan/Set5/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/Set5 --save_imgs  --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/Set14/LR/x4 -igt ../data/bsrgan/Set14/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/Set14 --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/Urban100/LR/x4 -igt ../data/bsrgan/Urban100/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/Urban100 --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/B100/LR/x4 -igt ../data/bsrgan/B100/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/B100 --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/bsrgan_plus/Manga109/LR/x4 -igt ../data/bsrgan_plus/Manga109/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/bsrgan_plus_d/"$2"/Manga109 --save_imgs --save_metrics
