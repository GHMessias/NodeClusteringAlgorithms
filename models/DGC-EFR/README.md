# DGC-EFR
Deep Graph Clustering with Enhanced Feature Representations for Community Detection 
## Requirements
Python = 3.7, PyTorch == 1.8.1, torchvision == 0.9.1
## Dataset
Dataset and pre-trained models can be downloaded here https://pan.baidu.com/s/1f-QHHiJYCdB6A8KkBX45LA, extraction code: 0oyx

To run code successfully, please unzip the dataset and place it in the current directory. 
## Pre-train AE and GAE models
python preae.py --name acm --epochs 50 --n_clusters 3
python pregae.py --name acm --epochs 50 --n_cluters 3
## Train our DGC-EFR model
python dgc_efr.py --name acm --epochs 200
