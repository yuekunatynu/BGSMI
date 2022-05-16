# Mutual Information Based Bayesian Graph Neural Network for Few-shot Learning

## Abstract
In the deep neural network based few-shot learning, the limited training data may make the neural network extract ineffective features, which leads to inaccurate results. By bayesian graph neural network (BGNN), the probability distributions on hidden layers imply useful features, and the correlation
among features could be used to improve the few-shot learning algorithms. Furthermore, the BGNN based few-shot learning could be improved by establishing the correlation among features. Thus, in this paper, we incorporate mutual information (MI) into BGNN to describe the correlation, and propose an innovative framework by adopting the Bayesian network with continuous variables (BNCV) for effective calculation of MI. First, we build the BNCV simultaneously when calculating the probability distributions of features from the Dropout in hidden layers of BGNN. Then, we approximate the MI values efficiently by probabilistic inferences over BNCV. Finally, we give the MI based loss function and training algorithm of our BGNN model. Experimental results show that our MI based BGNN framework is effective for few-shot learning and outperforms some state-of-the-art competitors by improving the accuracy with several orders of magnitude.

## Usage
```python Train.py```
## Cite

```
@inproceedings{
BGSMI,
title="{Mutual Information Based Bayesian Graph Neural Network for Few-shot Learning}",
author={},
booktitle={},
year={2022},
}
```
