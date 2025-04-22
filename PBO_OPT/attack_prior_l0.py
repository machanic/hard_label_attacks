import glob
import sys
import os
sys.path.append(os.getcwd())
import argparse
import json
import os
import os.path as osp
import random
from types import SimpleNamespace
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from config import IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from PBO.bo_l0 import BayesOpt

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
    def __init__(self, model, data, target, epsilon=8.0/255.0, original_size=32, reduce_size=None, is_targeted=False, nograd=True, mode='bilinear'):
        self.model = model.to(data.device)
        self.original_size = original_size
        self.reduce_size = reduce_size if reduce_size is not None else original_size
        self.is_targeted = is_targeted
        self.data = data
        self.target = target
        # self.epsilon = epsilon
        self.nograd = nograd
        self.mode = mode

        self.lamda = 10


    # def fine_grained_binary_search(self, x0, vector):
    #     """
    #     Finds the minimum n such that setting the smallest n values in the vector to zero activates f using binary search.
    #
    #     Args:
    #         vector (torch.Tensor): The input tensor of shape (3, 224, 224), with values in the range [-1, 1].
    #         f (callable): A function that takes a tensor as input and returns 0 or 1.
    #
    #     Returns:
    #         int: The smallest n such that f(thresholded_vector) == 1, or -1 if no such n exists.
    #     """
    #     query = 0
    #     while self.model(torch.clamp(x0 + vector.sign(), 0., 1.)).max(1)[1] == self.target:
    #         query += 1
    #         vector = torch.randn_like(vector)
    #
    #     flattened = vector.flatten()
    #     sorted_indices = torch.argsort(flattened.abs())
    #     lbd_lo, lbd_hi = 0, flattened.numel()
    #     num = flattened.numel()
    #
    #     while lbd_hi - lbd_lo >= 10:
    #         lbd_mid = (lbd_lo + lbd_hi) // 2
    #         thresholded = flattened.clone()
    #         thresholded[sorted_indices[:num-lbd_mid]] = 0
    #         reshaped = thresholded.view_as(vector)
    #         # print(f"Step: {query}, Non-zero elements: {torch.sum(thresholded != 0)}, Mid: {lbd_mid}, Low: {lbd_lo}, High: {lbd_hi}")
    #
    #         query += 1
    #         if self.model(torch.clamp(x0 + reshaped.sign(), 0., 1.)).max(1)[1] != self.target:
    #             lbd_hi = lbd_mid  # Search in the lower half
    #         else:
    #             lbd_lo = lbd_mid + 1  # Search in the upper half
    #
    #     # Debug info
    #     # print(f"Final Prediction: {self.model(torch.clamp(x0 + reshaped, 0., 1.)).max(1)[1]}, Target: {self.target}")
    #     # print(f"Final Non-zero Count: {torch.sum((torch.clamp(x0 + reshaped, 0., 1.) - x0) != 0)}, Mid: {lbd_mid}")
    #
    #     return torch.tensor(lbd_mid), query

    def fine_grained_binary_search(self, x0, vector):
        """
        Finds the minimum n such that setting the smallest n values in the vector to zero activates f using binary search.

        Args:
            vector (torch.Tensor): The input tensor of shape (3, 224, 224), with values in the range [-1, 1].
            f (callable): A function that takes a tensor as input and returns 0 or 1.

        Returns:
            int: The smallest n such that f(thresholded_vector) == 1, or -1 if no such n exists.
        """
        x = x0.clone().squeeze()
        query = 0
        while self.model(torch.clamp(x + vector.sign(), 0., 1.)).max(1)[1] == self.target:
            # query += 1
            # vector = torch.randn_like(vector)
            return 0

        # adv_x = torch.clamp(x + vector, 0., 1.)
        # numerator = (x * adv_x).sum(dim=0)  # 通道维度求和
        # x_norm = torch.sqrt((x ** 2).sum(dim=0))  # 图片的 L2 范数
        # mask_norm = torch.sqrt((adv_x ** 2).sum(dim=0))  # 掩码的 L2 范数
        # denominator = x_norm * mask_norm
        # denominator = torch.clamp(denominator, min=1e-8)
        # mask_cosine_similarity = numerator / denominator

        flattened = vector.flatten()
        sorted_indices = torch.argsort(flattened.abs())
        lbd_lo, lbd_hi = 0, flattened.numel() // 2
        num = flattened.numel()

        while lbd_hi - lbd_lo >= 10:
            lbd_mid = (lbd_lo + lbd_hi) // 2
            mask = torch.ones_like(flattened)
            mask[sorted_indices[:num-lbd_mid]] = 0
            mask = mask.view_as(vector)#.unsqueeze(0).repeat(3, 1, 1)
            # print(f"Step: {query}, Non-zero elements: {torch.sum(thresholded != 0)}, Mid: {lbd_mid}, Low: {lbd_lo}, High: {lbd_hi}")

            query += 1
            if self.model(torch.clamp(x + mask * vector.sign(), 0., 1.)).max(1)[1] != self.target:
                lbd_hi = lbd_mid  # Search in the lower half
            else:
                lbd_lo = lbd_mid + 1  # Search in the upper half

        # Debug info
        # print(f"Final Prediction: {self.model(torch.clamp(x0 + reshaped, 0., 1.)).max(1)[1]}, Target: {self.target}")
        # print(f"Final Non-zero Count: {torch.sum((torch.clamp(x0 + reshaped, 0., 1.) - x0) != 0)}, Mid: {lbd_mid}")

        return torch.tensor(lbd_mid), query


    def get_loss(self, pert):
        image_pert = pert.reshape([-1, 3, self.reduce_size, self.reduce_size])
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
            elif self.mode == 'nearest':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise
        res, nquery = [], 0
        for i in range(len(image_pert)):
            _res, query = self.fine_grained_binary_search(self.data, image_pert[i])
            res.append(_res)
            nquery += query
        return -torch.stack(res), nquery

    def __call__(self, pert):
        return self.get_loss(pert)


def run_bo(obj, bounds, n_iter, device='cuda', prior_obj=None, scale='adapt'):
    bo = BayesOpt(obj, bounds, device=device, n_opt=1, n_init=20, n_past=0,
                  normalize_y=True, exact_fval=True, prior_f=prior_obj, scale=scale)
    if scale != 'fixed_only':
        bo.initialize(bo.gen_rand_init())
        if bo.f.get_opt() > 0:
            log.info('Attack succeeded during initialization')
            obj.check_success(bo.f.get_opt_x())
            return bo.f.get_opt_x(), bo.f.call_count
    else:
        bo.initialize()
    x, _, _ = bo.run(n_iter=n_iter, freq_fit_hyper=10)
    if x is None:
        log.info('Attack failed')
    else:
        log.info('Attack succeeded')
        # obj.check_success(x)
    log.info('Total queries: {}'.format(bo.f.call_count))
    return x, bo.f.call_count


class PBO_attack(object):
    def __init__(self, args, surrogate_model):
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size, args.arch)
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.not_done_loss_all = torch.zeros_like(self.query_all)
        self.not_done_prob_all = torch.zeros_like(self.query_all)
        self.surrogate_model = surrogate_model

    def attack_all_images(self, args, arch_name, target_model, result_dump_path):
        reduce_size = args.size
        epsilon = args.epsilon
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            if batch_idx < args.start:
                continue
            if images.size(-1) != target_model.input_size[-1]:
                images = F.interpolate(images, size=target_model.input_size[-1], mode='bilinear', align_corners=False)
            selected = torch.arange(batch_idx * args.batch_size,
                                    min((batch_idx + 1) * args.batch_size, self.total_images))
            data, target = images.cuda(), true_labels.cuda()

            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                  size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logits = target_model(images)
                    target_labels = logits.argmin(dim=1).to(device)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset]).to(device)
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None

            assert data.shape[0] == 1 and target.shape[0] == 1
            with torch.no_grad():
                logit = target_model(data)
            pred = logit.argmax(dim=1)
            correct = pred.eq(target).float()

            if not correct.item():
                log.info(f'Image {batch_idx+1} is classified wrongly')
                continue
            dim = data.shape[0] * data.shape[1] * reduce_size * reduce_size
            bounds = [[-1, 1]] * dim

            if not args.targeted:
                log.info(f'Attack image {batch_idx+1}, original label {target.item()}')
                obj = Objective(target_model, data, target, original_size=target_model.input_size[-1], reduce_size=reduce_size,
                                epsilon=1, is_targeted=args.targeted, mode=args.dr, nograd=True)
                prior_obj = None
                if args.method == 'bo':
                    image_pert, query = run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)
            else:
                assert target.shape[0] == 1
                log.info(f'Attack image {batch_idx+1}, original label {target.item()}, new label {target_labels}')
                obj = Objective(target_model, data, target_labels, reduce_size=reduce_size,
                                epsilon=epsilon, is_targeted=args.targeted, mode=args.dr)
                prior_obj = None
                if args.method == 'bo':
                    image_pert, query = run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)
            # image_pert = torch.clamp(data + image_pert.reshape([-1, 3,32,32]) * 8/255, 0, 1)
            # print(torch.norm((image_pert-data).view(-1), p=np.inf))
            # if image_pert == None:
            #     image_pert = torch.zeros(dim).to(device)     # 攻击失败
            query = torch.tensor(query).float().cuda()
            with torch.no_grad():
                adv_logit = obj.get_logits(image_pert)
                adv_prob = F.softmax(adv_logit, dim=1)
            adv_pred = adv_logit.argmax(dim=1)
            if args.targeted:
                not_done = (1 - adv_pred.eq(target_labels).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = adv_pred.eq(target).float()
            success = (1 - not_done) * correct
            success_query = success * query
            not_done_prob = adv_prob[torch.arange(args.batch_size), target] * not_done
            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', 'not_done_prob']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()  # 由于value_all是全部图片都放在一个数组里，当前batch选择出来
            log.info("{}-th batch (size={}), current batch success rate:{:.3f}".format(batch_idx, image_pert.size(0),
                                                                                       success.mean().item()))
            exit()

def get_exp_dir_name(dataset, surrogate_arch, norm, targeted, target_type, args):
    from datetime import datetime
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if surrogate_arch is not None:
        if args.attack_defense:
            dirname = 'PBO_attack_on_defensive_model_{}_surrogate_arch_{}_{}_{}'.format(dataset, surrogate_arch, norm, target_str)
        else:
            dirname = 'PBO_attack_{}_surrogate_arch_{}_{}_{}'.format(dataset, surrogate_arch, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'PBO_attack_on_defensive_model_{}_{}_{}'.format(dataset, norm, target_str)
        else:
            dirname = 'PBO_attack_{}_{}_{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=1000)
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2|l0]')
    parser.add_argument('--json-config', type=str, default='../configures/PBO.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for bo attack.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str, default="")

    parser.add_argument('--method', default='bo', type=str, choices=['bo'],
                        help='to use bayesian optimization (bo)')
    parser.add_argument('--acq-function', default='UCB', type=str, choices=['UCB', 'EI'])
    parser.add_argument('--bo-scale', default='adapt', type=str, choices=['fixed', 'adapt', 'fixed_only'],
                        help='how to use surrogate models to guide optimization, see the doc in bo.py')
    parser.add_argument('--size', default=224, type=int, help='default=32: not using dimension reduction.'
                        'If <32, then reduce dimension to size*size*3')
    parser.add_argument('--start', type=int, default=0, help='skipping the first `start` images')
    parser.add_argument('--dr', type=str, default='nearest', choices=['bilinear', 'nearest'], help='the dimension reduction algorithm')


    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("using GPU {}".format(args.gpu))

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 20000
    args.surrogate_arch = None
    args.exp_dir = osp.join(args.exp_dir, get_exp_dir_name(args.dataset, args.surrogate_arch, args.norm,
                                                           args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                log_file_path = os.path.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                              args.defense_norm,
                                                                              args.defense_eps))
            else:
                log_file_path = os.path.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)

    surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False) if args.surrogate_arch is not None else None
    attacker = PBO_attack(args, surrogate_model)
    for arch in archs:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, arch, model, save_result_path)
        model.cpu()
