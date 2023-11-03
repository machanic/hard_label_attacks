import torch
import numpy as np
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.target_class_dataset import ImageNetDataset,CIFAR100Dataset,CIFAR10Dataset,TinyImageNetDataset
from config import PROJECT_PATH

def select_random_image_of_target_class(dataset_name, target_labels, target_model, load_random_class_image):
    images = []
    for label in target_labels:  # length of target_labels is 1
        if load_random_class_image:
            initial_images = np.load("{}/attacked_images/{}/{}_targeted-attack-initial-images.npz".format(PROJECT_PATH, dataset_name, dataset_name),
                            allow_pickle=True)
            image = torch.from_numpy(initial_images[str(label.item())]).unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                      size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                      align_corners=False)
        else:
            if dataset_name == "ImageNet":
                dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-10":
                dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "CIFAR-100":
                dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            elif dataset_name == "TinyImageNet":
                dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
            index = np.random.randint(0, len(dataset))
            image, true_label = dataset[index]
            image = image.unsqueeze(0)
            if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                image = F.interpolate(image,
                                      size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                      align_corners=False)
            with torch.no_grad():
                logits = target_model(image.cuda())
            max_recursive_loop_limit = 100
            loop_count = 0
            while logits.max(1)[1].item() != label.item() and loop_count < max_recursive_loop_limit:
                loop_count += 1
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and target_model.input_size[-1] != 299:
                    image = F.interpolate(image,
                                          size=(target_model.input_size[-2], target_model.input_size[-1]), mode='bilinear',
                                          align_corners=False)
                with torch.no_grad():
                    logits = target_model(image.cuda())

            if loop_count == max_recursive_loop_limit:
                # The program cannot find a valid image from the validation set.
                return None

            assert true_label == label.item()
        images.append(torch.squeeze(image))
    return torch.stack(images)  # B,C,H,W