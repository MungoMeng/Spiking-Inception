# Spiking Inception Architecture for Unsupervised Spiking Neural Networks (SNN)
Spiking Neural Network (SNN), as a brain-inspired machine learning algorithm, is attracting attention due to its 
event-driven computing style. Most unsupervised SNNs are trained through competitive learning with Spike-Timing-Dependent Plasticity (STDP). 
But previous SNNs trained through this approach are limited by slow learning speed and/or sub-optimal learning capability.
To ease these limitations. We proposed a Spiking Inception architecture for unsupervised SNN. 
Compared to widely used Fully-Connected (FC) and Locally-Connected (LC) architectures, STDP-based unsupervised SNNs using our architecture
have much improved learning capability, learning efficiency, and robustness.  
**For more details, please refer to our paper. [[Neucom](https://www.sciencedirect.com/science/article/abs/pii/S0925231221002733)] [[arXiv](https://arxiv.org/abs/2001.01680)].**

## Architecture
![architecture](https://github.com/MungoMeng/Spiking-Inception/blob/master/Figure/architecture.png)

The Inception module is proposed in the ANN literature. There is a Split-and-Merge strategy in the Inception module: 
The input is split into a few parallel pathways with a set of specialized filters (e.g. 3×3, 5×5, 7×7 convolutional kernels, pooling, etc.), 
and then all pathways merge by concatenation. Under this strategy, the Inception modules can integrate multi-scale spatial information 
and improve the network’s parallelism. We also designed an Inception-like multi-pathway network architecture. 
To further improve the architecture’s learning efficiency and robustness, we divided each pathway into multiple parallel 
sub-networks by partitioning competition areas. Finally we attained a high-parallelism Inception-like network architecture
consisting of 21 parallel sub-networks.   

## Instruction
Here we provide an implementation of our spiking Inception architecture. All code is written in **Python2**.

### Pre-reqirements
* Python-2.7
* Brian-2.2.1

Other versions of Brian (>=2.0) might work as well, but there is no guarantee on it.

### Train
Please `cd` to the directory containing all source code  such as `cd /path_to_src`, and then you can train the SNN with a simple commond:  
```
python Train.py
```
If it's the first time you run `Train.py`, you need to download the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset
and set the `MNIST_data_path` in the `Functions.py` to specify the directory containing the data. 
The trained weight file will be saved in a directory named `weights` (Sometimes you need to create it manually).

### Test
You need to set the `load_ending` in the `Test.py` to specify which weight file you want to load and test from `weights`. 
Then, you can test the trained SNN with a simple command:  
```
python Test.py
```
Note that running `Test.py` won't give you a testing result (accuracy) directly. 
It only saves the spiking activity in a directory named `activity` (Sometimes you need to create it manually).

### Evaluation
You can use the following command to get a testing result (accuracy) on the testing set of MNIST.
```
python Evaluation.py
```
Note that you need to set the `trained_sample` in the `Evaluation.py` to specify which activity file you want to load from `activity`.

## Citation
If this repository helps your work, please kindly cite our papers as follows:

* **Mingyuan Meng, Xingyu Yang, Lei Bi, Jinman Kim, Shanlin Xiao, Zhiyi Yu, "High-parallelism Inception-like Spiking Neural Networks for Unsupervised Feature Learning," Neurocomputing, 2021, doi: 10.1016/j.neucom.2021.02.027. [[Neucom](https://www.sciencedirect.com/science/article/abs/pii/S0925231221002733)] [[arXiv](https://arxiv.org/abs/2001.01680)]**
* **Mingyuan Meng, Xingyu Yang, Shanlin Xiao, Zhiyi Yu, "Spiking Inception Module for Multi-layer Unsupervised Spiking Neural Networks," 2020 International Joint Conference on Neural Networks (IJCNN), Glasgow, United Kingdom, 2020, pp. 1-8, doi: 10.1109/IJCNN48605.2020.9207161. [[IEEE](https://ieeexplore.ieee.org/document/9207161)] [[arXiv](https://arxiv.org/abs/2001.10696)]**
