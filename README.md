# Selfee: Self-supervised Features Extraction of animal behaviors

This is the official implementation of Selfee. In brief, Selfee is a **fully unsupervised** neural network for animal behavior analysis. **It is fast, sensitive, and unbiased.**

![Selfee](./img/selfee.jpg)
## Abstract
Fast and accurately characterizing animal behaviors is crucial for neuroscience research. Deep learning models are efficiently used in laboratories for behavior analysis. However, it has not been achieved to use a fully unsupervised method to extract comprehensive and discriminative features directly from raw behavior video frames for annotation and analysis purposes. Here, we report a self-supervised feature extraction (Selfee) convolutional neural network with multiple downstream applications to process video frames of animal behavior in an end-to-end way. Visualization and classification of the extracted features (Meta-representations) validate that Selfee processes animal behaviors in a comparable way of human understanding. We demonstrate that Meta-representations can be efficiently used to detect anomalous behaviors that are indiscernible to human observation and hint in-depth analysis. Furthermore, time-series analyses of Meta-representations reveal the temporal dynamics of animal behaviors. In conclusion, we present a self-supervised learning approach to extract comprehensive and discriminative features directly from raw video recordings of animal behaviors and demonstrate its potential usage for various downstream applications.

## Network structure

Selee is inspired by and modified from SimSiam and CLD:

https://github.com/facebookresearch/simsiam

https://github.com/frank-xwang/CLD-UnsupervisedLearning

![net](./img/network.jpg)

## Package availability 

Currently, we only provide unpolished code for the reproduction of our experiments. After publication, a python package would be available.

## Data Preprocessing

For data preprocessing, I recommend you to use the RodentTracker. It also provides other functions, such as animal tracking.

https://github.com/EBGU/RodentTracker
