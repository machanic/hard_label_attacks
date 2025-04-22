# -*- coding: gbk -*-
import argparse

import os
import random
import sys
sys.path.append(os.getcwd())
import json
from types import SimpleNamespace
import os.path as osp
import torch
import math
import numpy as np
import glog as log
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from utils.dataset_toolkit import select_random_image_of_target_class
from BASES.utils_bases import get_adv_np, get_label_loss

class BASES(object):
    def __init__(self, model, surrogate_victim_model, surrogate_models, dataset, clip_min, clip_max, height, width,
                 channels, norm, epsilon, load_random_class_image, total_images,
                 maximum_queries=10000, batch_size=1):
        """
        :param clip_min: lower bound of the image.
        :param clip_max: upper bound of the image.
        :param norm: choose between [l2, linf].
        :param iterations: number of iterations.
        :param gamma: used to set binary search threshold theta. The binary search
                     threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for linf attack.
        :param max_num_evals: maximum number of evaluations for estimating gradient.
        :param init_num_evals: initial number of evaluations for estimating gradient.
        """
        self.model = model
        self.surrogate_victim_model = surrogate_victim_model
        self.surrogate_models = surrogate_models
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        # self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (channels, height, width)
        self.load_random_class_image = load_random_class_image
        self.min_value = 0
        self.max_value = 1
        self.epsilon = math.sqrt(0.001 * np.prod(self.shape))

        self.n_iters = 10
        self.alpha = 3.0/255 * self.epsilon / self.n_iters
        self.lr_w = 5e-3
        self.fuse = 'loss'
        self.loss_name = 'cw'
        self.iterw = 50

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, model.arch)
        self.batch_size = batch_size
        self.total_images = total_images#len(self.dataset_loader.dataset)
        self.success_query_all = dict() # query times


    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1].detach().cpu().item() != true_labels[0].item()
        else:
            return logits.max(1)[1].detach().cpu().item() == target_labels[0].item()

    # def initialize(self, sample, target_images, true_labels, target_labels):
    #     """
    #     sample: the shape of sample is [C,H,W] without batch-size
    #     Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    #     """
    #     num_eval = 0
    #     if target_images is None:
    #         while True:
    #             random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float().cuda()
    #             # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
    #             success = self.decision_function(random_noise[None], true_labels, target_labels)
    #             num_eval += 1
    #             if success:
    #                 break
    #             if num_eval > 1000:
    #                 log.info("Initialization failed! Use a misclassified image as `target_image")
    #                 if target_labels is None:
    #                     target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
    #                                                   size=true_labels.size()).long().cuda()
    #                     invalid_target_index = target_labels.eq(true_labels)
    #                     while invalid_target_index.sum().item() > 0:
    #                         target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
    #                                                             size=target_labels[invalid_target_index].size()).long().cuda()
    #                         invalid_target_index = target_labels.eq(true_labels)
    #
    #                 initialization = select_random_image_of_target_class(self.dataset_name, target_labels, self.model, self.load_random_class_image).squeeze()
    #                 return initialization, 1
    #             # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
    #         # Binary search to minimize l2 distance to original image.
    #         low = 0.0
    #         high = 1.0
    #         while high - low > 0.001:
    #             mid = (high + low) / 2.0
    #             blended = (1 - mid) * sample + mid * random_noise
    #             success = self.decision_function(blended, true_labels, target_labels)
    #             num_eval += 1
    #             if success:
    #                 high = mid
    #             else:
    #                 low = mid
    #         # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very
    #         # small, this case will cause an infinity loop
    #         initialization = (1 - high) * sample + high * random_noise
    #     else:
    #         initialization = target_images
    #     return initialization, num_eval


    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        success_query = 0
        images = images.cuda()

        # x = images.cuda()
        # y = true_labels.cuda()
        # if target_labels is not None:
        #     target_labels = target_labels.cuda()
        #
        # x_adv, num_eval = self.initialize(x, target_images, y, target_labels)
        # # log.info("after initialize")
        # query += num_eval
        # dist = torch.norm((x_adv - x).view(batch_size, -1), self.ord, 1).cpu()
        # working_ind = torch.nonzero(dist > self.epsilon).view(-1)  # get locations (i.e., indexes) of non-zero elements of an array.
        # success_stop_queries[working_ind] = query[working_ind]  # success times
        #
        # for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
        #     self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        if target_labels is None:
            tgt_label = true_labels
        else:
            tgt_label = target_labels

        # start from equal weights
        w_np = np.array([1 for _ in range(len(self.surrogate_models))]) / len(self.surrogate_models)
        adv_np, losses = get_adv_np(images, tgt_label, w_np, self.surrogate_models, self.norm, self.epsilon, self.n_iters, self.alpha, fuse=self.fuse,
                                    untargeted=not args.targeted, loss_name=self.loss_name, adv_init=None)
        label_idx, _, _ = get_label_loss(adv_np, self.surrogate_victim_model, tgt_label, self.loss_name, targeted=args.targeted)
        n_query = 1
        loss_wb_list = losses  # loss of optimizing wb models

        success_query += 1
        if self.decision_function(adv_np, true_labels, target_labels):
            log.info('{}-th image, query: {}, attack success! '.format(batch_index, success_query))
            return success_query
        else:
            log.info('{}-th image, query: {}, attack failed! '.format(batch_index, success_query))

        idx_w = 0  # idx of wb in W

        while True:
            w_np_temp_plus = w_np.copy()
            w_np_temp_plus[idx_w] += self.lr_w
            adv_np_plus, losses_plus = get_adv_np(images, tgt_label, w_np_temp_plus, self.surrogate_models,
                                                  self.norm, self.epsilon, self.n_iters,
                                                  self.alpha, fuse=self.fuse, untargeted=not args.targeted,
                                                  loss_name=self.loss_name, adv_init=adv_np)
            label_plus, loss_plus, _ = get_label_loss(adv_np_plus, self.surrogate_victim_model, tgt_label, self.loss_name,
                                                      targeted=args.targeted)
            n_query += 1

            w_np_temp_minus = w_np.copy()
            w_np_temp_minus[idx_w] -= self.lr_w
            adv_np_minus, losses_minus = get_adv_np(images, tgt_label, w_np_temp_minus, self.surrogate_models,
                                                    self.norm, self.epsilon, self.n_iters,
                                                    self.alpha, fuse=self.fuse, untargeted=not args.targeted,
                                                    loss_name=self.loss_name, adv_init=adv_np)
            label_minus, loss_minus, _ = get_label_loss(adv_np_minus, self.surrogate_victim_model, tgt_label, self.loss_name,
                                                        targeted=args.targeted)
            n_query += 1

            # update
            if loss_plus < loss_minus:
                # loss = loss_plus
                w_np = w_np_temp_plus
                adv_np = adv_np_plus
                loss_wb_list += losses_plus
                last_idx = idx_w
            else:
                # loss = loss_minus
                w_np = w_np_temp_minus
                adv_np = adv_np_minus
                loss_wb_list += losses_minus
                last_idx = idx_w

            idx_w = (idx_w + 1) % len(self.surrogate_models)
            if n_query > 5 and last_idx == idx_w:
                self.lr_w /= 2  # decrease the lr
                # print(f"lr_w: {self.lr_w}")

            success_query += 1
            if self.decision_function(adv_np, true_labels, target_labels):
                log.info('{}-th image, query: {}, attack success! '.format(batch_index, success_query))
                return success_query
            else:
                log.info('{}-th image, query: {}, attack failed! '.format(batch_index, success_query))

            if success_query >= self.maximum_queries:
                return -1


    def attack_all_images(self, args, arch_name, result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if batch_index >= self.total_images:
                continue
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0: # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index+1))
                continue
            selected = torch.arange(batch_index * args.batch_size, min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted:
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[invalid_target_index].shape).long()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item()!=true_labels[0].item()
                    # log.info("load random label as {}".format(target_labels))
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                target_images = select_random_image_of_target_class(self.dataset_name, target_labels, self.model, self.load_random_class_image)
                if target_images is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index+1))
                    continue
            else:
                target_labels = None
                target_images = None

            success_query = self.attack(batch_index, images, target_images, true_labels, target_labels)
            self.success_query_all[batch_index] = success_query

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"success_query_all": self.success_query_all,
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'BASES_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'BASES-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, required=True)
    parser.add_argument('--json-config', type=str, default='./configures/Evolutionary.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-victim-arch', default=None, type=str, help='network architecture')
    parser.add_argument("--surrogate-archs", nargs="+", help="multiple surrogate models, and this parameter should be passed in through space splitting")
    parser.add_argument('--all-archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type', type=str, default='increment', choices=['random', "load_random", 'least_likely',"increment"])
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense_eps',type=str,default="")
    parser.add_argument('--max-queries',type=int, default=2)
    parser.add_argument('--total-images', type=int, default=1000)

    args = parser.parse_args()
    assert args.batch_size == 1, "Evolutionary Attack only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
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
    # if args.targeted:
    #     if args.dataset == "ImageNet":
    #         args.max_queries = 20000
    # if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
    #     args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))
    os.makedirs(args.exp_dir, exist_ok=True)

    if args.all_archs:
        if args.attack_defense:
            log_file_path = osp.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                              args.defense_norm,
                                                                              args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    surrogate_models = []
    for surrogate_arch in args.surrogate_archs:
        surrogate_model = StandardModel(args.dataset, surrogate_arch, False, load_pretrained=True)
        surrogate_model.cuda()
        surrogate_model.eval()
        surrogate_models.append(surrogate_model)

    surrogate_victim_model = StandardModel(args.dataset, args.surrogate_victim_arch, no_grad=True)
    surrogate_victim_model.cuda()
    surrogate_victim_model.eval()

    for arch in archs:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm,args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,
                                   norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker = BASES(model, surrogate_victim_model, surrogate_models, args.dataset, 0, 1.0,
                         model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                         args.norm, args.epsilon, args.load_random_class_image, args.total_images,
                         maximum_queries=args.max_queries, batch_size=args.batch_size)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()

