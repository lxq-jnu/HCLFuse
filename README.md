# [NeurIPS 2025 spotlight] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws.
This repository is the official implementation of the NeurIPS 2025 paper:"Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws".

# Framework
The overall framework of the proposed SpTFuse.
![image](https://github.com/lxq-jnu/HCLFuse/blob/main/images/framework.png)

# Abstract
Existing infrared and visible image fusion methods often face the dilemma of balancing modal information. Generative fusion methods reconstruct fused images by learning from data distributions, but their generative capabilities remain limited. Moreover, the lack of interpretability in modal information selection further affects the reliability and consistency of fusion results in complex scenarios. This manuscript revisits the essence of generative image fusion under the inspiration of human cognitive laws and proposes a novel infrared and visible image fusion method, termed HCLFuse. First, HCLFuse investigates the quantification theory of information mapping in unsupervised fusion networks, which leads to the design of a multi-scale mask-regulated variational bottleneck encoder. This encoder applies posterior probability modeling and information decomposition to extract accurate and concise low-level modal information, thereby supporting the generation of high-fidelity structural details. Furthermore, the probabilistic generative capability of the diffusion model is integrated with physical laws, forming a time-varying physical guidance mechanism that adaptively regulates the generation process at different stages, thereby enhancing the ability of the model to perceive the intrinsic structure of data and reducing dependence on data quality. Experimental results show that the proposed method achieves state-of-the-art fusion performance in qualitative and quantitative evaluations across multiple datasets and significantly improves semantic segmentation metrics. This fully demonstrates the advantages of this generative image fusion method, drawing inspiration from human cognition, in enhancing structural consistency and detail quality. 

## Recommended Environment
 - [ ] Python 3.10
 - [ ] torch 1.13.0
 - [ ] torchvision 0.14.0
 - [ ] kornia 0.7.3
 - [ ] pillow 10.4.0
 - [ ] numpy 1.24.4
 - [ ] scipy 1.10.1
 - [ ] tqdm 4.67.1

# To Train

Run ```python train.py``` to train the model.

# To Test

Download the checkpoint from [model](https://pan.baidu.com/s/1nY_nDf6Zv6csOPr6UDlgQQ?pwd=ot7j) and put it into `'./ckpt/'`.

Run `sample.py`.
