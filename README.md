# Three-Stream Temporal-Shift Attention Network Based on Self-Knowledge Distillation for Micro-Expression Recognition
## Introduction
Official code for "[Three-Stream Temporal-Shift Attention Network Based on Self-Knowledge Distillation for Micro-Expression Recognition](https://arxiv.org/abs/2406.17538)".
## Requirements
This repository is based on PyTorch 2.1.1, CUDA 11.8 and Python 3.10.13. All experiments in our paper were conducted on NVIDIA GeForce RTX 4090 GPU.
## Dataset
Put the micro-expression datasets in the `Dataset` folder. Take CASME II dataset preprocessing as an example:
1. Use `load_images.py` for face detection and cropping
2. Use `optflow_for_classify.py` for optical flow computation
3. Use `LOSO.py` for leave-one-subject-out cross-validation dataset splitting

## Usage
1. Configure `pre_trained_model_path`, `main_path`, and `exp_name` in `train_classify_SKD_TSTSAN.py`. We provide a pretrained SKD-TSTSAN model weight file trained on the macro-expression dataset in the `Pretrained_model` folder
2. Run `train_classify_SKD_TSTSAN.py` to train the model

## Citation

If our SKD-TSTSAN is useful for your research, please consider citing:

```bibtex
@article{zhu2024three,
  title={Three-stream temporal-shift attention network based on self-knowledge distillation for micro-expression recognition},
  author={Zhu, Guanghao and Liu, Lin and Hu, Yuhao and Sun, Haixin and Liu, Fang and Du, Xiaohui and Hao, Ruqian and Liu, Juanxiu and Liu, Yong and Deng, Hao and others},
  journal={arXiv preprint arXiv:2406.17538},
  year={2024}
}
```

## Acknowledgements
Our code uses [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface) for face detection. Please download the pretrained RetinaFace model file (`Resnet50_Final.pth`) from their repository and place it in the `RetinaFace` folder. We appreciate the valuable work of these authors.

## Questions
If you have any questions, welcome contact me at [gzhu663663@gmail.com](mailto:gzhu663663@gmail.com).
