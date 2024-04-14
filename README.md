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
This code heavily depends on the [BES](https://github.com/mt-cly/BES). If an error occurs in the train_cam or make_cam sections, please use [BES](https://github.com/mt-cly/BES) to generate CAM and place it in the corresponding folder for the subsequent steps.






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

1.If there is any bug or confusion, I am glad to discuss with you. 

2.If you encounter this problem: ValueError: could not convert string '2007_000032' to int32 at row 0, column 1. This is due to numpy version incompatibility. Older versions of NumPy are more tolerant of certain types of string conversions, or have stronger error-handling capabilities in certain cases. However, starting from newer versions of NumPy, it may be more strict, requiring strings to adhere to stricter formatting requirements when converting to integers. This can lead to type conversion errors in some cases, such as when strings contain underscores. We debugged using local tests and found that the issue does not occur with version 1.21.5, but does occur with version 1.24.4. There are two ways to address this: first, downgrade NumPy; second, modify the line img_name_list = np.loadtxt(dataset_path, dtype=np.int32) in the function def load_img_name_list(dataset_path) in the file voc/dataloader.py to ignore underscores () and change dtype=np.bool to dtype=np.bool in line 85 of misc/pyutils. We have already made these modifications, and you can choose the method that suits your situation.

如果你遇到这个问题：ValueError: could not convert string '2007_000032' to int32 at row 0, column 1。这是由于numpy版本不兼容导致的。较旧版本的NumPy对于某些类型的字符串转换更宽容，或者在某些情况下具有更强的容错能力。但是从较新版本的NumPy开始，它可能更加严格，要求字符串在转换为整数时必须满足更严格的格式要求。这可能会导致在某些情况下，例如字符串中包含下划线时，出现类型转换错误。我们使用本地测试进行调试，发现使用1.21.5版本时不会出现问题，而在使用1.24.4版本时则会出现此问题。你可以通过两种方法进行改进：一是降低NumPy的版本。二是在voc/dataloader.py文件的def load_img_name_list(dataset_path)函数中，将img_name_list = np.loadtxt(dataset_path, dtype=np.int32)这一行代码中进行修改，使其能够忽略下划线(_)，以及将misc/pyutils中的85行中的dtype=np.bool修改为dtype=np.bool\_，我们已经完成了修改，你可以根据自己情况选择方法。

3.You can install this package using the following command.：pip install git+https://github.com/lucasb-eyer/pydensecrf.git

4.本实验室的同学们如果使用DELL这台服务器跑时，无需修改任何路径直接运行即可。--2024.04.14

