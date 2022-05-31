# Selfee: Self-supervised Features Extraction of animal behaviors

This is the official implementation of [Selfee](https://www.biorxiv.org/content/10.1101/2021.12.24.474120v1). In brief, Selfee is a **fully unsupervised** neural network for animal behavior analysis. **It is fast, sensitive, and unbiased.**

<div align=center>
<img src=./img/selfee.jpg width=40%/>
</div>

## A tutorial of Selfee

1. Prepare a Conda environment for Selfee

```
conda env create -f Selfee.yml 
```
Selfee enviroment could support [RodentTracker](https://github.com/EBGU/RodentTracker), so you could skip this part when using RodentTracker.

2. Download pretrained weights from Google Drive

We provide [pretrained weights](https://drive.google.com/file/d/1A3U5guNEKA3Bi9H3QnfstZDEZ6aesqcR/view?usp=sharing) on flies or mice datasets via Google Drive.


## Abstract
Fast and accurately characterizing animal behaviors is crucial for neuroscience research. Deep learning models are efficiently used in laboratories for behavior analysis. However, it has not been achieved to use a fully unsupervised method to extract comprehensive and discriminative features directly from raw behavior video frames for annotation and analysis purposes. Here, we report a self-supervised feature extraction (Selfee) convolutional neural network with multiple downstream applications to process video frames of animal behavior in an end-to-end way. Visualization and classification of the extracted features (Meta-representations) validate that Selfee processes animal behaviors in a comparable way of human understanding. We demonstrate that Meta-representations can be efficiently used to detect anomalous behaviors that are indiscernible to human observation and hint in-depth analysis. Furthermore, time-series analyses of Meta-representations reveal the temporal dynamics of animal behaviors. In conclusion, we present a self-supervised learning approach to extract comprehensive and discriminative features directly from raw video recordings of animal behaviors and demonstrate its potential usage for various downstream applications.

## Network structure

Selee is inspired by and modified from [SimSiam](https://github.com/facebookresearch/simsiam) and [CLD](https://github.com/frank-xwang/CLD-UnsupervisedLearning):





![net](./img/network.jpg)

## Pretrained weight

We provide [pretrained weights](https://drive.google.com/file/d/1A3U5guNEKA3Bi9H3QnfstZDEZ6aesqcR/view?usp=sharing) on flies or mice datasets via Google Drive.


## Data Preprocessing

For data preprocessing, I recommend you to use my [RodentTracker](https://github.com/EBGU/RodentTracker). It also provides other functions, such as animal tracking.


