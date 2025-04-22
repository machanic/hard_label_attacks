import torch

from models import clip
import os
from torch import nn

from config import PROJECT_PATH, CLASS_NUM, IN_CHANNELS


class VisionLanguageModel(nn.Module):
    def __init__(self, dataset, arch, no_grad=True):
        super(VisionLanguageModel, self).__init__()
        self.dataset = dataset
        self.arch = arch
        self.num_classes = CLASS_NUM[dataset]
        self.no_grad = no_grad
        self.in_channels = IN_CHANNELS[dataset]

        if arch.startswith("CLIP"):
            self.imagenet_label_file_path = os.path.join(PROJECT_PATH, "train_pytorch_model", "CLIP",
                                                         "imagenet-labels.txt")
            self.clip_pretrained_model_file_path = os.path.join(PROJECT_PATH, "train_pytorch_model", "CLIP",
                                                                arch[arch.index("-")+1:] + ".pt")
            assert os.path.exists(self.imagenet_label_file_path), "{} does not exist!".format(
                self.imagenet_label_file_path)
            assert os.path.exists(self.clip_pretrained_model_file_path), "{} does not exist!".format(
                self.clip_pretrained_model_file_path)

            self.mean = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = self.std = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            model, preprocess = clip.load(self.clip_pretrained_model_file_path, device='cuda')
            self.input_range = [0, 1]
            self.input_size = [self.in_channels, model.visual.input_resolution, model.visual.input_resolution]
            self.input_space = 'RGB'
            self.model = model
            with open(self.imagenet_label_file_path, 'r') as file:
                labels = [line.strip().strip('"').split(',')[0] for line in file]
            self.text = clip.tokenize(labels).cuda()
            with torch.no_grad():
                self.text_features = self.model.encode_text(self.text)

    def get_logits(self, image):
        logits_per_image, logits_per_text = self.model(image.detach(), self.text_features.detach())
        return logits_per_image

    def forward(self, x):
        x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
        if self.no_grad:
            with torch.no_grad():
                logits = self.get_logits(x)
        else:
            with torch.enable_grad():
                logits = self.get_logits(x)
        return logits