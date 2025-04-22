import clip
import json


import torch
import time
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
args = parser.parse_args()
args.print_freq = 10

with open("./imagenet-simple-labels.json", 'r') as file_obj:
    imagenet_classes = json.load(file_obj)


dataset='ImageNet'


model, preprocess = clip.load("ViT-B/32", device=device)
model.cuda()

val_dataset = ImageFolder(root='D:/dataset/ILSVRC2012/validation/',transform=preprocess)
batch_size = 50
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,)

with torch.no_grad():
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_classes]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, 5)):
    """计算指定Top k值的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 获取每个样本的前maxk个预测
        _, pred = output.topk(maxk, 1, True, True)  # pred shape: [batch_size, maxk]
        pred = pred.t()  # pred shape: [maxk, batch_size]
        target = target.view(1, -1).expand_as(pred)  # target shape: [maxk, batch_size]

        correct = pred.eq(target)  # correct shape: [maxk, batch_size]

        res = []
        for k in topk:
            # 对每个样本的前k个预测进行判断
            correct_k = correct[:k].any(dim=0).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model,  args):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            image_features = model.encode_image(input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T

            acc1, acc5 = accuracy(logits, target, topk=(1,5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    top1=top1, top5=top5))
        print('Validate Set Acc@1 {top1.avg:.3f}'.format(top1=top1))
        # print('Validate Set Acc@5 {top5.avg:.3f}'.format(top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    validate(val_loader=val_loader, model=model,  args=args)
