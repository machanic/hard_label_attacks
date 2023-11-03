import torch
import time
import torch.nn as nn
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.standard_model import StandardModel

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
args = parser.parse_args()
args.print_freq = 10

dataset='ImageNet'
# arch = "jx"
# arch = "T2TViT"
# arch = "swin"
# arch = "gcvit"
arch = "pnasnet5large"

standard_model = StandardModel(dataset=dataset, arch=arch)
model = standard_model.make_model(dataset=dataset, arch=arch, in_channels=3, num_classes=1000, trained_model_path=None, load_pretrained=True)
model.cuda()

transform = transforms.Compose([transforms.Resize((model.input_size[-2], model.input_size[-1])), transforms.ToTensor()])
val_dataset = ImageFolder(root='D:/download/ImageNet ILSVRC 2012 validation/ILSVRC2012/validation', transform=transform)
batch_size = 50
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k]
            correct_k = correct_k.reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate_accuracy mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} {top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
        print('Validate Set Acc@1 {top1.avg:.3f}'.format(top1=top1))
        # print('Validate Set Acc@5 {top5.avg:.3f}'.format(top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    image_classifier_loss = nn.CrossEntropyLoss().cuda()
    validate(val_loader=val_loader, model=model, criterion=image_classifier_loss, args=args)
