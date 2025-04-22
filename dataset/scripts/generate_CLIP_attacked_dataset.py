import argparse
import glob
import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import glog as log
from config import PROJECT_PATH, MODELS_TRAIN_STANDARD, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.multimodal_model import VisionLanguageModel
from torch.nn import functional as F



def generate_attacked_dataset(dataset, num_sample, models):
    selected_images = []
    selected_true_labels = []
    selected_img_id = []
    total_count = 0
    data_loader = DataLoaderMaker.get_imgid_img_label_data_loader(dataset, 100, is_train=False, shuffle=True)
    log.info("begin select")
    if dataset != "ImageNet":
        for idx, (image_id, images, labels) in enumerate(data_loader):
            log.info("read {}-th batch images".format(idx))
            images_gpu = images.cuda()
            print(images.size())
            pred_eq_true_label = []
            for model in models:
                model.cuda()
                with torch.no_grad():
                    logits = model(images_gpu)
                model.cpu()
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            current_select_count = len(np.where(pred_eq_true_label)[0])
            total_count += current_select_count
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            if total_count >= num_sample:
                break
    else:
        for idx, (image_id, images, labels) in enumerate(data_loader):
            log.info("read {}-th batch images".format(idx))
            pred_eq_true_label = []
            for model in models:
                if model.input_size[-1] != 299:
                    images_gpu = F.interpolate(images, size=(model.input_size[-2], model.input_size[-1]),
                                               mode='bilinear', align_corners=False)
                    images_gpu = images_gpu.cuda()  # 3 x 299 x 299
                else:
                    images_gpu = images.cuda()
                with torch.no_grad():
                    model.cuda()
                    logits = model(images_gpu)
                    model.cpu()
                pred = logits.max(1)[1]
                correct = pred.detach().cpu().eq(labels).long()
                pred_eq_true_label.append(correct.detach().cpu().numpy())
            pred_eq_true_label = np.stack(pred_eq_true_label).astype(np.uint8) # M, B
            pred_eq_true_label = np.bitwise_and.reduce(pred_eq_true_label, axis=0)  # 1,0,1,1,1
            current_select_count = len(np.where(pred_eq_true_label)[0])
            total_count += current_select_count
            selected_image = images.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]]
            selected_images.append(selected_image)
            selected_true_labels.append(labels.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            selected_img_id.append(image_id.detach().cpu().numpy()[np.where(pred_eq_true_label)[0]])
            log.info("We have selected {} images".format(total_count))
            if total_count >= num_sample:
                break
    selected_images = np.concatenate(selected_images, 0)
    selected_true_labels = np.concatenate(selected_true_labels, 0)
    selected_img_id = np.concatenate(selected_img_id, 0)

    selected_images = selected_images[:num_sample]
    selected_true_labels = selected_true_labels[:num_sample]
    selected_img_id = selected_img_id[:num_sample]
    return selected_images, selected_true_labels, selected_img_id

def save_selected_images(selected_images, selected_true_labels, selected_img_id, save_path):
    np.savez(save_path, images=selected_images, labels=selected_true_labels, image_id=selected_img_id)

def load_models(dataset):
    archs = ["resnet50","convit_base","crossvit_base_224","maxvit_rmlp_small_rw_224","jx_vit", "gcvit_base", "swin_base_patch4_window7_224"]
    model_path_list = []

    if dataset == "CIFAR-10" or dataset == "CIFAR-100":
        for arch in MODELS_TEST_STANDARD[dataset]:
            test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                PROJECT_PATH, dataset, arch)
            if os.path.exists(test_model_path):
                archs.append(arch)
                model_path_list.append(test_model_path)
            else:
                log.info(test_model_path + " does not exist!")
    else:
        log.info("begin check arch")
        for arch in archs:
            test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/hub/checkpoints/{}*.pth".format(
                PROJECT_PATH, dataset, arch)
            test_model_path = list(glob.glob(test_model_list_path))
            if len(test_model_path) == 0:  # this arch does not exists in args.dataset
                log.info(test_model_list_path + " does not exist!")
                continue
            model_path_list.append(test_model_path[0])
            log.info("check arch {} done, archs length {}".format(arch, len(model_path_list)))

    log.info("begin construct model")
    clip_model1 = StandardModel(dataset, "CLIP-ViT-L-14", True)
    clip_model1.cuda()
    clip_model1.eval()
    # log.info("load CLIP over from {}".format(clip_model1.clip_pretrained_model_file_path))

    clip_model2 = StandardModel(dataset, "CLIP-ViT-L-14-336px", True)
    clip_model2.cuda()
    clip_model2.eval()
    # log.info("load CLIP over from {}".format(clip_model2.clip_pretrained_model_file_path))

    # clip_model3 = StandardModel(dataset, "CLIP-ViT-B-32", True)
    # clip_model3.cuda()
    # clip_model3.eval()
    # # log.info("load CLIP over from {}".format(clip_model3.clip_pretrained_model_file_path))
    #
    # clip_model4 = StandardModel(dataset, "CLIP-ViT-B-16", True)
    # clip_model4.cuda()
    # clip_model4.eval()
    # # log.info("load CLIP over from {}".format(clip_model4.clip_pretrained_model_file_path))
    #
    #
    # clip_model5 = StandardModel(dataset, "CLIP-RN50x64", True)
    # clip_model5.cuda()
    # clip_model5.eval()
    # # log.info("load CLIP over from {}".format(clip_model5.clip_pretrained_model_file_path))
    #
    # clip_model6 = StandardModel(dataset, "CLIP-RN50x16", True)
    # clip_model6.cuda()
    # clip_model6.eval()
    # # log.info("load CLIP over from {}".format(clip_model6.clip_pretrained_model_file_path))

    # clip_model7 = VisionLanguageModel(dataset, "CLIP-RN50x4", True)
    # clip_model7.cuda()
    # clip_model7.eval()
    # log.info("load CLIP over from {}".format(clip_model7.clip_pretrained_model_file_path))
    #
    # clip_model8 = VisionLanguageModel(dataset, "CLIP-RN50", True)
    # clip_model8.cuda()
    # clip_model8.eval()
    # log.info("load CLIP over from {}".format(clip_model8.clip_pretrained_model_file_path))
    #
    # clip_model9 = VisionLanguageModel(dataset, "CLIP-RN101", True)
    # clip_model9.cuda()
    # clip_model9.eval()
    # log.info("load CLIP over from {}".format(clip_model9.clip_pretrained_model_file_path))

    models = [clip_model1, clip_model2, ]
    for arch in archs:
        model = StandardModel(dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        models.append(model)
        log.info("load {} over".format(arch))
    log.info("end construct model")
    return models

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"])
    args = parser.parse_args()
    dataset = args.dataset
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    save_path = "{}/attacked_images/{}/{}_images_for_CLIP.npz".format(PROJECT_PATH, dataset, dataset)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_log_file(os.path.dirname(save_path)+"/generate_{}.log".format(dataset))
    models = load_models(dataset)
    selected_images, selected_true_labels, selected_img_id = generate_attacked_dataset(dataset, 1000, models)

    save_selected_images(selected_images, selected_true_labels, selected_img_id, save_path)
    print("Saved to {}".format(save_path))
    print("done")
