[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Annalina-Luo/ClipNews/blob/main/ClipNews_training.ipynb)

# ClipNews

## Prerequisites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation

- Clone this repo:
```bash
git clone https://github.com/UIC-ESLAS/DiGAN-pytorch
cd NewsClip
```

- For pip users, please type the command `pip install -r requirements.txt`.
- For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

## Dataset Preparation
- Download the datasets and training the model in [Colab](https://colab.research.google.com/github/Annalina-Luo/ClipNews/blob/main/ClipNews_training.ipynb)
- A quick way to acquire the dataset is sending a email to annalinaluo@gmail.com

## Traing a model
```bash
python main.py --batch_size=64 --TextEncoder_attention=True
```
- To continue training, append `--checkpoint="./checkpoint/checkpoint_ClipNews.pth.tar` on the command line.

## Inference
