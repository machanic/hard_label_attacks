import os
import sys
import argparse
from torch.nn import functional as F

import torch
import numpy as np
import glog as log
from config import PROJECT_PATH, IMAGE_DATA_ROOT, CLASS_NUM
from models.standard_model import StandardModel
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset

MODELS = {"CIFAR-10": ["alexnet", "densenet-bc-100-12", "densenet-bc-L190-k40", "preresnet-110","resnext-16x64d",
                       "resnext-8x64d","vgg19_bn","resnet-20","resnet-32","resnet-44","resnet-50","resnet-56",
                       "resnet-110","resnet-1202","pyramidnet272", "WRN-28-10-drop","WRN-40-10-drop",
                       "densenet-bc-L190-k40","gdas"],
          "ImageNet": ["alexnet", "bninception","densenet121", "densenet161","densenet169", "densenet201","dpn68",
                       "resnext101_32x4d","resnext101_64x4d","se_resnext101_32x4d","se_resnext50_32x4d",
                       "squeezenet1_0","squeezenet1_1","vgg11","vgg11_bn","vgg13_bn","vgg13","vgg16",
                       "vgg16_bn","vgg19_bn","vgg19","resnext101_64x4d","inceptionv4","senet154","resnet101",
                       "inceptionv3","pnasnet5large"]}

def load_models(dataset_name):
    archs = MODELS[dataset_name]
    models = []
    for arch in archs:
        log.info("Load: {}".format(arch))
        model = StandardModel(dataset_name, arch, no_grad=True)
        models.append(model)
    return models

def test(model, image, label, dataset_name):
    if dataset_name == "ImageNet" and model.input_size[-1] != 299:
        image = F.interpolate(image,
                               size=(model.input_size[-2], model.input_size[-1]), mode='bilinear',
                               align_corners=False)
    model.eval()
    with torch.no_grad():
        pred = model(image).max(1)[1].detach().cpu().item() == label
    model.cpu()
    return pred

def main(dataset_name):
    target_images = {}
    models = load_models(dataset_name)
    for target_label in range(CLASS_NUM[dataset_name]):
        if dataset_name == "CIFAR-10":
            dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], target_label, "train")
        elif dataset_name == "ImageNet":
            dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name], target_label, "train")

        success = []
        while sum(success) < len(models):
            image = dataset[np.random.randint(0, len(dataset))][0]
            success = list(map(lambda model: test(model.cuda(), image[None].cuda(), target_label, dataset_name), models))
            log.info('{}-th images, success / total: {}/{}'.format(target_label, sum(success), len(success)))
        target_images[str(target_label)] = image

    np.savez('{}/attacked_images/{}/{}_targeted-attack-initial-images.npz'.format(PROJECT_PATH, dataset_name, dataset_name), **target_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    np.random.seed(0)
    main(args.dataset)
