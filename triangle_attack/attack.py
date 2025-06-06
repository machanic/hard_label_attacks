import argparse

import os
import random
import sys
sys.path.append(os.getcwd())
from collections import defaultdict, OrderedDict

import json
from types import SimpleNamespace
import os.path as osp
import torch
import numpy as np
import glog as log
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IMAGE_DATA_ROOT, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from utils.dataset_toolkit import select_random_image_of_target_class

from attack_mask import get_x_hat_in_2d, get_difference, get_orthogonal_1d_in_subspace
import triangle_attack.torch_dct as torch_dct

class TriangleAttack(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, batch_size, epsilon, norm,
                 dim_num, side_length, max_iter_num_in_2d, plus_learning_rate, minus_learning_rate, half_range, init_alpha,
                 load_random_class_image, maximum_queries=10000):
        self.model = model
        self.batch_size = batch_size
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, model.arch)
        self.images = torch.from_numpy(self.dataset_loader.dataset.images).cuda()
        self.labels = torch.from_numpy(self.dataset_loader.dataset.labels).cuda()
        self.total_images = len(self.dataset_loader.dataset)
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.shape = (channels, height, width)

        self.dim_num = dim_num
        self.side_length = side_length
        self.max_iter_num_in_2d = max_iter_num_in_2d
        self.plus_learning_rate = plus_learning_rate
        self.minus_learning_rate = minus_learning_rate
        self.half_range = half_range
        self.init_alpha = init_alpha
        self.load_random_class_image = load_random_class_image
        self.maximum_queries = maximum_queries
        self.dataset_name = dataset

        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)  # 查询次数
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.linf_distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1].detach().cpu().item() != true_labels[0].item()
        else:
            return logits.max(1)[1].detach().cpu().item() == target_labels[0].item()

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(
                    np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float().cuda()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise[None], true_labels, target_labels)
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0,
                                                                                high=CLASS_NUM[self.dataset_name],
                                                                                size=target_labels[
                                                                                    invalid_target_index].size()).long().cuda()
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = select_random_image_of_target_class(self.dataset_name, target_labels,
                                                                          self.model, self.load_random_class_image).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise[None]
                success = self.decision_function(blended, true_labels, target_labels)
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very
            # small, this case will cause an infinity loop
            initialization = (1 - high) * sample + high * random_noise
        else:
            initialization = target_images
        return initialization, num_eval

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        batch_size = images.size(0)
        images = images.squeeze().cuda()
        true_labels = true_labels.cuda()

        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1) * self.batch_size, self.total_images)).tolist()
        x_adv, num_evals = self.initialize(images, target_images, true_labels, target_labels)
        x_adv = x_adv.cuda()
        query += num_evals
        dist = torch.norm((x_adv - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                inside_batch_index].item()

        # # global probability
        # probability = np.ones(images.shape[1] * images.shape[2])
        # # global is_visited_1d
        # is_visited_1d = torch.zeros(images.shape[0] * images.shape[1] * images.shape[2])
        # # global selected_h
        # # global selected_w
        # selected_h = images.shape[1]
        # selected_w = images.shape[2]
        get_x_hat_in_2d.alpha = np.pi / 2

        get_x_hat_in_2d.total = 0
        get_x_hat_in_2d.clamp = 0

        images = images[np.newaxis, :, :, :]
        original_label = true_labels.reshape(1, )
        # if target_images is not None:
        #     target_images = target_images.squeeze()
        # Initialize. Note that if the original image is already classified incorrectly, the difference between the found initialization and sample is very very small, this case will lead to inifinity loop later.
        while query < self.maximum_queries:

            x_o2x_adv = torch_dct.dct_2d(get_difference(images, x_adv))
            axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv)
            direction, mask = get_orthogonal_1d_in_subspace(self.side_length, x_o2x_adv, self.dim_num, args.ratio_mask, args.dim_num)
            axis_unit2 = direction / torch.norm(direction)
            x_adv, num_eval, changed = get_x_hat_in_2d(torch_dct.dct_2d(images), torch_dct.dct_2d(x_adv), axis_unit1,
                                                      axis_unit2, self.model, original_label,
                                                      max_iter=self.max_iter_num_in_2d,
                                                      plus_learning_rate=self.plus_learning_rate,
                                                      minus_learning_rate=self.minus_learning_rate,
                                                      half_range=self.half_range, init_alpha=self.init_alpha)
            query += num_eval

            dist = torch.norm((x_adv - images).contiguous().view(batch_size, -1), self.ord, 1)
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
            for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

            log.info('{}-th image, {}: distortion {:.10f}, query: {}'.format(batch_index+1, self.norm, dist.item(), int(query[0].item())))
            if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                break

        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return x_adv, query, success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name, result_dump_path):
        assert self.norm == 'l2'
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            with torch.no_grad():
                logit = self.model(images.cuda())
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index + 1))
                continue
            selected = torch.arange(batch_index * args.batch_size,
                                    min((batch_index + 1) * args.batch_size, self.total_images))
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
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
            else:
                target_labels = None
                target_images = None

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels)
            adv_images = adv_images.detach().cpu()

            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
            linf_distortion_with_max_queries = torch.norm((adv_images - images).contiguous().view(args.batch_size, -1), np.inf, 1).detach().cpu()
            with torch.no_grad():
                if adv_images.dim() == 3:
                    adv_images = adv_images.unsqueeze(0)
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.detach().cpu() * (
                        success_query.detach().cpu() <= self.maximum_queries).float()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries", "linf_distortion_with_max_queries"]:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()

        log.info('{} is attacked finished ({} images)'.format(arch_name, self.total_images))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "avg_not_done": self.not_done_all[self.correct_all.bool()].mean().item(),
                          "mean_query": self.success_query_all[self.success_all.bool()].mean().item() if self.success_all.sum().item() > 0 else 0,
                          "median_query": self.success_query_all[self.success_all.bool()].median().item() if self.success_all.sum().item() > 0 else 0,
                          "max_query": self.success_query_all[self.success_all.bool()].max().item() if self.success_all.sum().item() > 0 else 0,
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "not_done_all": self.not_done_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_all": self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "linf_distortion_with_max_queries": self.linf_distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'TriangleAttack_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'TriangleAttack-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/TriangleAttack.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all-archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type',type=str, default='increment', choices=['random', "load_random", 'least_likely',"increment"])
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense_eps',type=str,default="")
    parser.add_argument('--max-queries',type=int, default=10000)
    # parser.add_argument('--save-linf-norm', action="store_true")

    parser.add_argument('--dim-num', type=int, default=3, help='the number of picked dimensions')
    parser.add_argument('--ratio-mask',type=float)
    parser.add_argument('--max-iter-num-in-2d',type=int,default=2,help='the maximum iteration number of attack algorithm in 2d subspace')
    parser.add_argument('--init-theta',type=int,default=2,help='the initial angle of a subspace=init_theta*np.pi/32')
    parser.add_argument('--init-alpha',type=float,default=np.pi / 2,help='the initial angle of alpha')
    parser.add_argument('--plus-learning-rate',type=float,default=0.01,help='plus learning_rate when success')
    parser.add_argument('--minus-learning-rate',type=float,default=0.0005,help='minus learning_rate when fail')
    parser.add_argument('--half-range',type=float,default=0.1,help='half range of alpha from pi/2')

    args = parser.parse_args()
    assert args.batch_size == 1, "Triangle Attack only supports mini-batch size equals 1!"
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
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
        args.max_queries = 20000
    args.exp_dir = osp.join(args.exp_dir,
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
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
        # if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
        #     for arch in MODELS_TEST_STANDARD[args.dataset]:
        #         test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
        #             PROJECT_PATH,
        #             args.dataset, arch)
        #         if os.path.exists(test_model_path):
        #             archs.append(arch)
        #         else:
        #             log.info(test_model_path + " does not exists!")
        # elif args.dataset == "TinyImageNet":
        #     for arch in MODELS_TEST_STANDARD[args.dataset]:
        #         test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
        #             root=PROJECT_PATH, dataset=args.dataset, arch=arch)
        #         test_model_path = list(glob.glob(test_model_list_path))
        #         if test_model_path and os.path.exists(test_model_path[0]):
        #             archs.append(arch)
        # else:
        #     for arch in MODELS_TEST_STANDARD[args.dataset]:
        #         test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
        #             PROJECT_PATH,
        #             args.dataset, arch)
        #         test_model_list_path = list(glob.glob(test_model_list_path))
        #         if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
        #             continue
        #         archs.append(arch)
    else:
        assert args.arch is not None
        archs = [args.arch]

    args.arch = ", ".join(archs)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
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
        args.side_length = model.input_size[-1]
        attacker = TriangleAttack(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                  args.batch_size, args.epsilon, args.norm,
                                  args.dim_num, args.side_length, args.max_iter_num_in_2d, args.plus_learning_rate,
                                  args.minus_learning_rate, args.half_range, args.init_alpha, args.load_random_class_image,
                                  maximum_queries=args.max_queries)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
