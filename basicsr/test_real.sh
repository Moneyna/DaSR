#! /bin/bash
#e.g. ./basicsr/test_real.sh pretrain_model DaSR_pretrain true 256 256 8 3 64 128 256 0 6 0 256 0.0 0 4

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/RealSR/LR/x4 -igt ../data/RealSR/HR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/real/"$2"/RealSR --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/RealSRSet/LR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/real/"$2"/RealSRSet --save_imgs --save_metrics

python basicsr/inference.py -params $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15}  -i ../data/DPED/LR -w ../experiments/"$1"/net_g_best_.pth -o ./Stage2Exp/reexam/real/"$2"/DPED --save_imgs --save_metrics
