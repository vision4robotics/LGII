# Exploiting Local-Global Image Illuminator for Nighttime UAV Tracking

This project inclueds code and demo videos of LGII.

# Abstract 
Object trackers have been deployed on unmanned aerial vehicles (UAVs) to expand enormous UAV-based autonomous applications. However, inevitable nighttime environments
slow down the promising expansion. In nighttime UAV applications, head-scratching low illumination not only impede human operators to initialize target objects but also hinder state-of-the-art trackers to extract valuable features. To dispel the darkness for both operators and trackers, this work proposes a novel local-global image illuminator, i.e., LGII. Specifically, a nested pyramid network is constructed to realize pixel-level feature enhancement. The dual-illumination decoder with multi-head attention is employed to facilitate global operator perception. Moreover, driven by a set of well-designed loss functions, LGII is trained with an unpaired nighttime UAV tracking dataset to cope with specific nighttime UAV challenges, e.g., fast motion, illumination variation, and occlusion. To verify the advantage of LGII in assisting both operators and trackers, image enhancement benchmarks and the public UAVDark135 benchmark are applied together. Evaluations of these benchmarks demonstrate that LGII facilitates operator perception and tracking performance obviously. In real-world tests, LGII assists the SOTA tracker to achieve real-time and robust nighttime tracking.


# Contact 
Haolin Dong

Email: 1851146@tongji.edu.cn

Changhong Fu

Email: changhongfu@tongji.edu.cn

# Demonstration running instructions

### Requirements

1.Python 3.7.10

2.Pytorch 1.10.1

4.torchvision 0.11.2

5.cuda 11.3.1

>Download the package, extract it and follow two steps:
>
>1. Put test images in data/test_data/, put training data in data/train_data/.
>
>2. For testing, run:
>
>     ```
>     python lowlight_test.py
>     ```
>     You can find the enhanced images in data/result/. Some examples have been put in this folder.
>   
>3. For training, run:
>
>     ```
>     python lowlight_train.py
>     ```



# Acknowledgements

We sincerely thank the contribution of `Chongyi Li` for his previous work Zero-DCE (https://github.com/Li-Chongyi/Zero-DCE).
