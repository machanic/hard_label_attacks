# Introduction
This repository includes the source code for "Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors".
To further support reproducibility, we provide the complete attack code for our method and all the baseline methods in [https://github.com/machanic/hard_label_attacks](https://github.com/machanic/hard_label_attacks).

# Repository Structure
```
+-- configures
|   |-- PriorOPT.json  # the hyperparameters setting of Prior-OPT and Prior-Sign-OPT.
+-- dataset
|   |-- dataset_loader_maker.py  # it returns the data loader class that includes 1000 attacks images for the experiments.
|   |-- npz_dataset.py  # it is the dataset class that includes 1000 attacks images for the experiments.
+-- models
|   |-- defensive_model.py # the wrapper of defensive networks (e.g., AT, ComDefend, Feature Scatter), and it converts the input image's pixels to the range of 0 to 1 before feeding.
|   |-- standard_model.py # the wrapper of standard classification networks, and it converts the input image's pixels to the range of 0 to 1 before feeding.
+-- PriorOPT # the folder of Prior-OPT and Prior-Sign-OPT. 
|   |-- attack.py  # the code of the attack with undefended surrogate models (i.e., the normal models without adversarial training as the surrogate models).
|   |-- attack_for_defense.py # the code of the attack with defense surrogate models (e.g., adversarially trained models (AT models))
|   |-- prior_opt_l2_norm_attack.py # the code of the L2 norm attack's process of Prior-OPT and Prior-Sign-OPT
|   |-- prior_opt_linf_norm_attack.py # the code of the L-infinity norm attack's process of Prior-OPT and Prior-Sign-OPT
|-- config.py   # the main configuration.
|-- logs  # all the output (logs and result stats files) are located inside this folder
|-- train_pytorch_model  # the pretrained weights of target models
|-- attacked_images  # the 1000 image data for evaluation 
```
The folder of `attacked_images` contains the 1000 tested images, which are packaged into `.npz` format with the pixel range of `[0-1]`.

The folder of `train_pytorch_model` contains the pretrained weights of target models.

Due to file size limitations, the folders of `attacked_images` and `train_pytorch_model` can be downloaded from [https://github.com/machanic/hard_label_attacks](https://github.com/machanic/hard_label_attacks).

All logs and results are saved in the `logs` folder, with results stored in `.json` format.

# Attack Command

The following command runs Prior-OPT on the ImageNet dataset under the untargeted attack setting.
Note that you can add the `--sign` argument to run Prior-Sign-OPT.

## Single surrogate model commands:
```
nohup python PriorOPT/attack.py --dataset ImageNet --arch resnet101  --gpu 0 --norm l2 --surrogate-arch resnet50 > /dev/null 2>&1 &
nohup python PriorOPT/attack.py --dataset ImageNet --arch inceptionv3  --gpu 0 --norm l2 --surrogate-arch inceptionresnetv2 > /dev/null 2>&1 &
```

## Multiple surrogate models commands:
```
nohup python PriorOPT/attack.py --dataset ImageNet --arch swin_base_patch4_window7_224  --gpu 2 --norm l2 --surrogate-archs resnet50 jx_vit > /dev/null 2>&1 &
nohup python PriorOPT/attack.py --dataset ImageNet --arch gcvit_base  --gpu 4 --norm l2 --surrogate-archs resnet50 jx_vit > /dev/null 2>&1 &
```

## Attacks against defense models commands:
```
nohup python PriorOPT/attack.py  --attack-defense --defense-model adv_train_on_ImageNet --defense-norm l2 --defense-eps 3 --arch resnet50 --dataset ImageNet --norm l2 --epsilon 10.0  --gpu 0  --surrogate-archs resnet50 senet154 resnet101 > /dev/null 2>&1 &
nohup python PriorOPT/attack.py  --attack-defense --defense-model adv_train_on_ImageNet --defense-norm linf --defense-eps 4_div_255 --arch resnet50 --dataset ImageNet --norm l2 --epsilon 10.0  --gpu 2 --surrogate-archs resnet50 senet154 resnet101 > /dev/null 2>&1 &
nohup python PriorOPT/attack.py  --attack-defense --defense-model adv_train_on_ImageNet --defense-norm linf --defense-eps 8_div_255 --arch resnet50 --dataset ImageNet --norm l2 --epsilon 10.0  --gpu 0 --surrogate-archs resnet50 senet154 resnet101 > /dev/null 2>&1 &
```

## Targeted attacks commands:
```
nohup python PriorOPT/attack.py --dataset ImageNet --arch swin_base_patch4_window7_224  --gpu 0 --norm l2 --targeted --load-random-class-image --surrogate-archs resnet50 jx_vit > /dev/null 2>&1 &
nohup python PriorOPT/attack.py --dataset ImageNet --arch resnet101  --gpu 0 --norm l2 --targeted --load-random-class-image --surrogate-arch resnet50 --sign > /dev/null 2>& 1&
nohup python PriorOPT/attack.py --dataset ImageNet --arch senet154  --gpu 0 --norm l2 --targeted --load-random-class-image --surrogate-arch resnet50 --sign > /dev/null 2>& 1&
nohup python PriorOPT/attack.py --dataset ImageNet --arch inceptionv3  --gpu 1 --norm l2 --targeted --load-random-class-image --surrogate-arch inceptionresnetv2 --sign > /dev/null 2>& 1&
nohup python PriorOPT/attack.py --dataset ImageNet --arch inceptionv4  --gpu 2 --norm l2 --targeted --load-random-class-image --surrogate-arch inceptionresnetv2 --sign > /dev/null 2>& 1&
nohup python PriorOPT/attack.py --dataset ImageNet --arch resnext101_64x4d  --gpu 0 --norm l2 --targeted --load-random-class-image --surrogate-arch resnet50 --sign > /dev/null 2>& 1&
```

Once the attack is running, logs are written to a newly created `logs` folder. After the attack, the statistical results are also saved in the same folder as `*.json` files.

* The gpu device could be specified by the `--gpu device_id` argument.
* the targeted attack can be specified by the `--targeted` and `--load-random-class-image` arguments. For an untargeted attack, omit this argument.
* For defense model attacks, use the `--attack_defense --defense_model adv_train/TRADES` argument.

# Requirement
Our code has been tested on the following environment (it should work on other environments with minimal changes):

* Ubuntu 18.04
* Python 3.7.3
* CUDA 11.1
* CUDNN 8.0.4
* PyTorch 1.7.1
* torchvision 0.8.2
* numpy 1.18.0
* pretrainedmodels 0.7.4
* bidict 0.18.0
* advertorch 0.1.5
* glog 0.3.1

You can install the required packages by running:

`pip install -r requirements.txt`

# Download Files of Pre-trained Models and Running Results
In summary, there are three additional folders that you can download or train by yourself: `attacked_images`, `train_pytorch_model`, and optionally `logs`.
