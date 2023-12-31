# SBE-main
>paper: Salient-Boundary-Guided Pseudo-Pixel Supervision for Weakly-Supervised Semantic Segmentation
>
In this paper, we propose a novel Saliency-guided Boundary Extraction (SBE) framework for supervising WSSS. Our SBE approach employs saliency maps(SMs) to guide the object boundary detection and the attention of CAMs towards foreground regions during the propagation of the coarse localization maps, resulting in highquality pixel-level pseudo masks. The pseudo masks can be further serve as category labels for supervising an off-the-shelf semantic segmentation network such as the DeepLab-v2.

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
 | Dataset | mIoU(val) | mIoU(test) |
| --- | --- | --- |
| PASCAL VOC	 | 71.7 | 71.2 |
| MS COCO	 | 39.1 | --- |


## citation
If you use our codes and models in your research, please cite:
>@ARTICLE{10363373,
  author={Shi, Min and Deng, Weizhao and Yi, Qingming and Liu, Weiping and Luo, Aiwen},
  journal={IEEE Signal Processing Letters}, 
  title={Salient-Boundary-Guided Pseudo-Pixel Supervision for Weakly-Supervised Semantic Segmentation}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/LSP.2023.3343945}}


## PS

If there is any bug or confusion, I am glad to discuss with you. 

