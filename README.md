# Salient-Boundary-Guided Pseudo-Pixel Supervision for Weakly-Supervised Semantic Segmentation
>[Salient-Boundary-Guided Pseudo-Pixel Supervision for Weakly-Supervised Semantic Segmentation](https://ieeexplore.ieee.org/document/10363373)
>
In this paper, we propose a novel Saliency-guided Boundary Extraction (SBE) framework for supervising WSSS. Our SBE approach employs saliency maps(SMs) to guide the object boundary detection and the attention of CAMs towards foreground regions during the propagation of the coarse localization maps, resulting in highquality pixel-level pseudo masks. The pseudo masks can be further serve as category labels for supervising an off-the-shelf semantic segmentation network such as the DeepLab-v2.

## abstract
This article presents an innovative approach for
generating pixel-wise pseudo masks as supervision for imagelevel
Weakly Supervised Semantic Segmentation (WSSS). This
is achieved by leveraging abundant object boundaries extracted
with the guidance of saliency maps (SMs). Initially, we synthesize
the boundary labels by combining Class Activation Maps (CAMs)
and SMs. Then, an elaborately-designed joint training strategy is
employed to fully exploit the complementary relationship between
the foreground of CAMs and the background and boundary of
SMs to yield rich object boundaries. Finally, we refine the CAMs
based on the constraints imposed by the extracted boundaries,
leading to more accurate pixel-wise pseudo masks.We thoroughly
evaluate the performance of our proposed pseudo masks through
extensive experiments, demonstrating their effectiveness as the
supervision for accurate semantic segmentation. Specifically, our
method achieves 71.7% mIoU and 39.1% mIoU on the validation
sets of PASCAL VOC 2012 and MS COCO 2014, respectively.

## framework

![Overview of our proposed SBE framework, including three core components: boundary label generation by synergistically leveraging class activation maps (CAMs) and saliency maps (SMs), boundary detection guided by SMs, and pseudo mask generation by refining CAMs using the extracted boundaries.](https://github.com/DL3399/SBE/blob/main/1703999985389.jpg)
Overview of our proposed SBE framework, including three core components: boundary label generation by synergistically leveraging class activation maps (CAMs) and saliency maps (SMs), boundary detection guided by SMs, and pseudo mask generation by refining CAMs using the extracted boundaries.


## USAGE
This code heavily depends on the [BES](https://github.com/mt-cly/BES). 
#### Preparation
* Dataset: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) & [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
* Python: 3.6
* Others: python3.6-dev, etc
#### Install python packages
```python
 pip install -r requirement.txt
```
For pytorch, torchvision, etc, installation command can be easily found with specified setting on [official website](https://pytorch.org/get-started/locally/).Here we use pytorch 1.8.

#### Run the code
Specify the VOC dataset path and run the command
```python
python run_sample.py --voc12_root xxxxx
```

## perfermance
![pseudo masks](https://github.com/DL3399/SBE/blob/main/1704000054495.jpg)
Qualitative comparison for pseudo masks: (a) Inputs, (b) GT, (c)
SEAM [16], (d) EPS [22], (e) RCA [34], (f) BES [19] and (g) our SBE.
 | Dataset | mIoU(val) | mIoU(test) |
| --- | --- | --- |
| PASCAL VOC	 | 71.7 | 71.2 |
| MS COCO	 | 39.1 | --- |


## citation
If you use our codes and models in your research, please cite:
>
M. Shi, W. Deng, Q. Yi, W. Liu and A. Luo, "Salient-Boundary-Guided Pseudo-Pixel Supervision for Weakly-Supervised Semantic Segmentation," in IEEE Signal Processing Letters, vol. 31, pp. 86-90, 2024, doi: 10.1109/LSP.2023.3343945.


## PS

If there is any bug or confusion, I am glad to discuss with you. 

