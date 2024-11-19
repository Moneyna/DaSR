# DaSR

This is the official PyTorch codes for the paper:
**Degradation-antagonistic Codebook Regression Network for Blind Image Super-Resolution**

## Dependencies and Installation

- Ubuntu >= 18.04
- CUDA >= 11.0
- Other required packages in `requirements.txt`
```
# create new anaconda env
conda create -n DaSR python=3.10
source activate DaSR

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Train SR model

```
# for Stage Ⅰ
python basicsr/train.py -opt options/train_dasr_HQ_stage.yml

# for Stage Ⅱ pre-training
python basicsr/train.py -opt options/train_dasr_bicubic_stage.yml

# for Stage Ⅱ
python basicsr/train.py -opt options/train_dasr_LQ_stage.yml

```

## Test SR model

### Dataset
Please prepare the testing datasets Set5, Set14, B100, Urban100, Manga109 and DIV2K Val and use [`make_datasets/make_test_bsrgan.py`]() to generate LR images.
```
python make_test_bsrganPlus.py
```
### Test shell
The pretrain model is put in the ```experiments/pretrain_model``` directory.
```
# test on synthetic datasets
./basicsr/test_bsrganPlus_degradation.sh pretrain_model DaSR_pretrain true 256 256 8 3 64 128 256 0 6 0 256 0.0 0 4

# test on real-world datasets
./basicsr/test_real.sh pretrain_model DaSR_pretrain true 256 256 8 3 64 128 256 0 6 0 256 0.0 0 4
```

## Ablation Study
```
# The visualization of Fig.6 in Main paper 
python attn_score_print.py -params true 256 256 8 3 64 128 256 6 0 256 0 4 -i ../data/bsrgan_plus/Set5/LR/x4 -igt ../data/bsrgan_plus/Set5/HR -w ../experiments/DaSR_LQ_stage_256_256_wGAN/models/net_g_best_.pth -o ./attn_map_result/DaSR/Set5
```

## Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR)
and [FeMaSR](https://github.com/chaofengc/FeMaSR).