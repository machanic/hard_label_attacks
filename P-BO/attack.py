import torch
import numpy as np
import argparse
from models import *
import torchvision
import torchvision.transforms as transforms
from bo import BayesOpt


def cw_loss(logits, targets, is_targeted=False, num_classes=10):
    onehot_targets = torch.zeros([targets.size(0), num_classes]).to(targets.device)
    onehot_targets[np.arange(len(targets)), targets] = 1.0

    target_logits = torch.sum(onehot_targets * logits, dim=1)
    other_logits = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

    if not is_targeted:
        loss = other_logits - target_logits
    else:
        loss = target_logits - other_logits

    return loss


class Objective:
    def __init__(self, model, data, target, epsilon=8.0/255.0, original_size=32, reduce_size=None, is_targeted=False, nograd=False, mode='bilinear'):
        self.model = model.to(data.device)
        self.original_size = original_size
        self.reduce_size = reduce_size if reduce_size is not None else original_size
        self.is_targeted = is_targeted
        self.data = data
        self.target = target
        self.epsilon = epsilon
        self.nograd = nograd
        self.mode = mode

    def get_logits(self, pert):
        image_pert = pert.reshape([-1, 3, self.reduce_size, self.reduce_size])
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
            elif self.mode == 'nearest':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise
        image = torch.clamp(self.data + image_pert * self.epsilon, 0, 1)
        res = self.model(image)
        return res.detach() if self.nograd else res

    def get_loss(self, pert):
        res = cw_loss(self.get_logits(pert), self.target, is_targeted=self.is_targeted)
        return res.detach() if self.nograd else res

    def __call__(self, pert):
        return self.get_loss(pert)

    def get_grad(self, pert):
        if self.nograd:
            raise NotImplementedError("Cannot get grad if self.nograd is True")
        x = pert.detach()
        x.requires_grad_()
        obj = self.get_loss(x)
        obj.backward()
        return x.grad.detach()

    def predict(self, pert):
        res = self.get_logits(pert).max(dim=1)[1]
        return res.detach() if self.nograd else res

    def check_success(self, pert):
        pred = self.predict(pert)
        if not self.is_targeted:
            assert all(pred != self.target)
        else:
            assert all(pred == self.target)


def run_bo(obj, bounds, device='cuda', prior_obj=None, scale='adapt'):
    bo = BayesOpt(obj, bounds, device=device, n_opt=1, n_init=20, n_past=0,
                  normalize_y=True, exact_fval=True, prior_f=prior_obj, scale=scale)
    if scale != 'fixed_only':
        bo.initialize(bo.gen_rand_init())
        if bo.f.get_opt() > 0:
            print('Attack succeeded during initialization')
            obj.check_success(bo.f.get_opt_x())
            return
    else:
        bo.initialize()
    x, _, _ = bo.run(n_iter=1000)
    if x is None:
        print('Attack failed')
    else:
        print('Attack succeeded')
        obj.check_success(x)
    print('Total queries:', bo.f.call_count)


class Sampler:
    def __init__(self, reduce_size, original_size, mode='bilinear'):
        self.reduce_size = reduce_size
        self.original_size = original_size
        self.mode = mode

    def __call__(self):
        rnd = torch.randn([1, 3, self.reduce_size, self.reduce_size])
        if self.mode == 'bilinear':
            rnd = torch.nn.functional.interpolate(rnd, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
        elif self.mode == 'nearest':
            rnd = torch.nn.functional.interpolate(rnd, [self.original_size, self.original_size], mode='nearest-exact')
        else:
            raise
        rnd = rnd.flatten()
        return rnd


def gen_model(model_str, model_path, device):
    if model_str is None:
        return None
    model_dict = {
        'resnet50': ResNet50(),
        'densenet121': DenseNet121(),
        'senet': SENet18(),
        'wideresnet': WideResNet(34)
    }
    net = model_dict[model_str]
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    model_path = f'cifar_ckpts/{model_str}_best.pt' if model_path is None else model_path
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack CIFAR10 models')
    parser.add_argument('--model', type=str, help='model', choices=['densenet121', 'resnet50', 'senet', 'wideresnet'])
    parser.add_argument('--model-path', default=None, type=str, help='model path')
    parser.add_argument('--surrogate-model', type=str, default=None, help='surrogate model', choices=['densenet121', 'resnet50', 'senet', 'wideresnet'])
    parser.add_argument('--surrogate-model-path', default=None, type=str, help='surrogate model path')
    parser.add_argument('--method', default='bo', type=str, choices=['bo'],
                        help='to use bayesian optimization (bo)')
    parser.add_argument('--bo-scale', default='adapt', type=str, choices=['fixed', 'adapt', 'fixed_only'],
                        help='how to use surrogate models to guide optimization, see the doc in bo.py')
    parser.add_argument('--size', default=32, type=int, help='default=32: not using dimension reduction.'
                        'If <32, then reduce dimension to size*size*3')
    parser.add_argument('--start', type=int, default=0, help='skipping the first `start` images')
    parser.add_argument('--count', type=int, default=10000, help='attack how many images')
    parser.add_argument('--dr', type=str, default='nearest', choices=['bilinear', 'nearest'], help='the dimension reduction algorithm')

    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = gen_model(args.model, args.model_path, device)
    prior_net = gen_model(args.surrogate_model, args.surrogate_model_path, device)

    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=1)

    num_classes = 10
    epsilon = 8.0 / 255.0
    if args.original_obj and args.method == 'zoo':
        reduce_size = 32
        sampler = Sampler(reduce_size=args.size, original_size=32, mode=args.dr)
    else:
        reduce_size = args.size
        sampler = None
    is_targeted = False
    total_count = 0

    for idx, (data, target) in enumerate(testloader):
        if idx < args.start:
            continue
        data, target = data.to(device), target.to(device)
        assert data.shape[0] == 1 and target.shape[0] == 1
        label = target.item()
        classification = net(data).max(dim=1)[1].item()
        if label != classification:
            print(f'Image {idx} is classified wrongly')
            continue
        dim = data.shape[0] * data.shape[1] * reduce_size * reduce_size
        bounds = [[-1, 1]] * dim

        if not is_targeted:
            print(f'Attack image {idx}, original label {label}')
            obj = Objective(net, data, target, reduce_size=reduce_size,
                            epsilon=epsilon, is_targeted=is_targeted, mode=args.dr)
            prior_obj = Objective(prior_net, data, target, reduce_size=reduce_size,
                                  epsilon=epsilon, is_targeted=is_targeted, mode=args.dr) if prior_net else None
            if args.method == 'bo': 
                run_bo(obj, bounds, device=device, prior_obj=prior_obj, scale=args.bo_scale)
        else:
            assert target.shape[0] == 1
            for i in range(num_classes):
                if i == label:
                    continue
                print(f'Attack image {idx}, original label {label}, new label {i}')
                new_target = torch.tensor([i]).to(device)
                obj = Objective(net, data, new_target, reduce_size=reduce_size,
                                epsilon=epsilon, is_targeted=is_targeted, mode=args.dr)
                prior_obj = Objective(prior_net, data, new_target, reduce_size=reduce_size,
                                      epsilon=epsilon, is_targeted=is_targeted, mode=args.dr) if prior_net else None
                if args.method == 'bo': 
                    run_bo(obj, bounds, device=device, prior_obj=prior_obj, scale=args.bo_scale)
        total_count += 1
        if total_count >= args.count:
            break
