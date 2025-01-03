## GLDM
**Paper**: Zero-shot Dynamic MRI Reconstruction with   Global-to-local Diffusion Model
**Authors**: Yu Guan, Kunlong Zhang, Qi Qi, Dong Wang, Ziwen Ke, Shaoyu Wang, Dong Liang, Qiegen Liu* 
NMR in Biomedicine
https://arxiv.org/abs/2411.03723

Date : January-3-2025  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Mathematics and Computer Sciences, Nanchang University. 


Diffusion models have emerged as promising tools for tackle the challenges of MRI reconstruction, demonstrat-ing superior performance in sample generation com-pared to traditional methods. However, their application in dynamic MRI reconstruction remains relatively un-derexplored, primarily owing to the substantial demand for fully-sampled training data, which is challenging to obtain because of the spatiotemporal complexity and high acquisition costs associated with dynamic MRI. To address this challenge, this paper proposes a zero-shot learning framework for accurate dynamic MR image reconstruction from under-sampled k-space data directly. Specifically, a unique time-interleaved acquisition scheme is employed to merge under-sampled k-space data from adjacent temporal frames, thereby constructing pseudo fully-encoded reference data. Moreover, while merging all the frames enhances the signal-to-noise ratio (SNR), it also reduces inter-frame correlation. In contrast, merging only local adjacent frames preserves in-ter-frame uniqueness but decreases the SNR. Therefore, a two-stage refinement strategy is applied during the diffu-sion process to learn the global-to-local prior, ensuring the diffusion model effectively captures the data distribu-tion for zero-shot reconstruction. Extensive experiments demonstrate that the proposed method performs well in terms of noise reduction and detail preservation, achiev-ing reconstruction quality comparable to that of super-vised approaches.

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Training Demo
``` bash
python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result
```
## Test Demo
``` bash
python PCsampling_demo_svd.py
```

## Graphical representation
### The whole pipeline of GLDM is illustrated in fig_1
<div align="center"><img src="https://github.com/yqx7150/GLDM/blob/main/fig_1.png" >  </div>
The schematic of the proposed GLDM algorithm. Red and blue parts represent the training stage that fully encoded full-resolution reference data is constructed through a time-interleaved acquisition scheme. Red part merges all time frames to train the global model (GM) while the blue part merges local time frames to train the local model (LM). Green part represents the reconstruction stage which the structure of the reconstruction model exists in a cascade form and the under-sampled k-space data (16 frames) are sequentially input into the network. At the same time, optimization unit (OU) containing a LR operator and a DC term is introduced to better remove aliasing and restore details






