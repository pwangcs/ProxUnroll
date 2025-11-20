# [Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging]([https://dl.acm.org/doi/abs/10.1145/3581783.3612242](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Proximal_Algorithm_Unrolling_Flexible_and_Efficient_Reconstruction_Networks_for_Single-Pixel_CVPR_2025_paper.html))

CVPR 2025 [[Arxiv](https://arxiv.org/abs/2505.23180)]
 
[Ping Wang](https://scholar.google.com/citations?user=WCsIUToAAAAJ&hl=zh-CN&oi=ao), [Lishun Wang](https://scholar.google.com/citations?user=BzkbrCgAAAAJ&hl=zh-CN&oi=sra), Gang Qu, [Xiaodong Wang](https://scholar.google.com/citations?user=2JXMfrcAAAAJ&hl=zh-CN&oi=sra), [Yulun Zhang](https://scholar.google.com/citations?user=ORmLjWoAAAAJ&hl=zh-CN), [Xin Yuan](https://scholar.google.com/citations?user=cS9CbWkAAAAJ&hl=zh-CN)

## Abstract
Deep-unrolling and plug-and-play (PnP) approaches have become the de-facto standard solvers for single-pixel imaging (SPI) inverse problem. PnP approaches, a class of iterative algorithms where regularization is implicitly performed by an off-the-shelf deep denoiser, are flexible for varying compression ratios (CRs) but are limited in reconstruction accuracy and speed. Conversely, unrolling approaches, a class of multi-stage neural networks where a truncated iterative optimization process is transformed into an endto-end trainable network, typically achieve better accuracy with faster inference but require fine-tuning or even retraining when CR changes. In this paper, we address the challenge of integrating the strengths of both classes of solvers. To this end, we design an efficient deep image restorer (DIR) for the unrolling of HQS (half quadratic splitting) and ADMM (alternating direction method of multipliers). More importantly, a general proximal trajectory (PT) loss function is proposed to train HQS/ADMM-unrolling networks such that learned DIR approximates the proximal operator of an ideal explicit restoration regularizer. Extensive experiments demonstrate that, the resulting proximal unrolling networks can not only flexibly handle varying CRs with a single model like PnP algorithms, but also outperform previous CR-specific unrolling networks in both reconstruction accuracy and speed.


<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/summary.png"  width="800"> 

 TL;DR: ProxUnroll achieves SOTA performance with high flexibility and fast convergence.
</div>

## ProxUnroll

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/proxunroll.png"  width="800">
 
  Proximal algorithm unrolling via trajectory loss.
</div>

<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/network.png"  width="800"> 

 Deep image restorer $\mathcal{R_{\theta}}$ used in ProxUnroll.
</div>

## Result
<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/result.png" width="800">
</div>
<div align="center">
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/simulated_visualization.png" width="800">
</div>
<div align="center"> 
  <img src="https://github.com/pwangcs/ProxUnroll/blob/main/fig/real_visualization.png" width="800">
</div>

## Citation
If you use ProxUnroll, please consider citing:
```
@inproceedings{wang2023saunet,
  title={Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging},
  author={Wang, Ping and Wang, Lishun and Qu, Gang and Wang, Xiaodong and Zhang, Yulun and Yuan, Xin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={411--421},
  year={2025}
}
```

## Contact

If you have any question, please contact wangping@westlake.edu.cn
