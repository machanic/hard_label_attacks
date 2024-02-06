#coding=utf-8
import argparse

import os
import random
import sys

from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from collections import defaultdict, OrderedDict

import json
from types import SimpleNamespace
import os.path as osp
import torch
import numpy as np
import glog as log
import copy
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from dataset.target_class_dataset import ImageNetDataset,CIFAR100Dataset,CIFAR10Dataset,TinyImageNetDataset
from utils.dataset_toolkit import select_random_image_of_target_class

class PAR(object):
    def __init__(self, model, dataset,  clip_min, clip_max, height, width, channels, norm, load_random_class_image, epsilon,iterations=40,
                 max_num_evals=1e4, init_num_evals=100,maximum_queries=10000,batch_size=1,random_direction=False):
        """
        :param clip_min: lower bound of the image.
        :param clip_max: upper bound of the image.
        :param norm: choose between [l2, linf].
        :param iterations: number of iterations.
        :param gamma: used to set binary search threshold theta. The binary search
                     threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for linf attack.
        :param stepsize_search: choose between 'geometric_progression', 'grid_search'.
        :param max_num_evals: maximum number of evaluations for estimating gradient.
        :param init_num_evals: initial number of evaluations for estimating gradient.
        """
        self.model = model
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.dim = height * width * channels
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (channels, height, width)
        self.load_random_class_image = load_random_class_image

        self.init_num_evals = init_num_evals
        self.max_num_evals = max_num_evals
        self.num_iterations = iterations

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images) # 查询次数
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.random_direction = random_direction


    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()

        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1].detach().cpu() != true_labels
        else:
            return logits.max(1)[1].detach().cpu() == target_labels

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(
                    np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise, true_labels, target_labels)
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long()
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0,
                                                                                high=CLASS_NUM[self.dataset_name],
                                                                                size=target_labels[
                                                                                    invalid_target_index].size()).long()
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = select_random_image_of_target_class(self.dataset_name, target_labels, self.model,
                                                                         self.load_random_class_image).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
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

    def clip_image(self, image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
        return torch.min(torch.max(image, clip_min), clip_max)


    def l2_distance(self,a, b):
        if type(b) != torch.Tensor:
            b = torch.ones_like(a).cuda() * b

        dist = (torch.sum((torch.round(a) / 255.0 - torch.round(b) / 255.0) ** 2)) ** 0.5

        return dist

    # def normalize_noise(self,direction, distance, original_image):
    #     norm_direction = direction / self.l2_distance(direction, 0)  # 归一化
    #
    #     clipped_direction = torch.clip(torch.round(norm_direction * distance + original_image), 0, 255) - original_image
    #
    #     clipped_dist = self.l2_distance(clipped_direction, 0)
    #
    #     return clipped_direction, clipped_dist

    # def scatter_draw(self,data):
    #     save_path = "/home/syc/adversarial_machine_learning/nips18-avc-attack-template__/"
    #     fig = plt.figure(figsize=(16, 9))
    #     plt.scatter(data[1], data[0], s=1)
    #     plt.savefig(save_path + "data.png", bbox_inches='tight')

    # def clip(self,x, min_x=-1, max_x=1):
    #     x[x < min_x] = min_x
    #     x[x > max_x] = max_x
    #     return x

    def value_mask_init(self,patch_num):  # 初始化查询价值mask
        value_mask = torch.ones([patch_num, patch_num]).cuda()
        # value_mask[int(patch_num*0.25):int(patch_num*0.75) , int(patch_num*0.25):int(patch_num*0.75)] = 0.5

        return value_mask

    def noise_mask_init(self,x, image, patch_num, patch_size):  # 初始化噪声幅度mask

        noise = x - image
        noise_mask = torch.zeros([patch_num, patch_num]).cuda()
        for row_counter in range(patch_num):
            for col_counter in range(patch_num):
                noise_mask[row_counter][col_counter] = self.l2_distance(
                    noise[(row_counter * patch_size):(row_counter * patch_size + patch_size),
                    (col_counter * patch_size):(col_counter * patch_size + patch_size)], 0)


        return noise_mask

    def translate(self,index, patch_num):  # 将价值最高patch的行列输出出来
        best_row = index // patch_num
        best_col = index - patch_num * best_row

        return best_row, best_col

    # def predictions(self, inputs):
    #
    #     logits = self.model.forward(np.round(inputs).astype(np.float32))
    #     return np.argmax(logits), logits

    def distance(self, input1, input2, min_, max_):
        return np.mean((input1 - input2) ** 2) / ((max_ - min_) ** 2)

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()

        assert images.size(0) == 1
        batch_size = images.size(0)
        images = images.squeeze()

        if target_images is not None:
            target_images = target_images.squeeze()
        # Initialize. Note that if the original image is already classified incorrectly, the difference between the found initialization and sample is very very small, this case will lead to inifinity loop later.
        perturbed, num_eval = self.initialize(images, target_images, true_labels, target_labels)
        # log.info("after initialize")
        query += num_eval
        dist = torch.norm((perturbed - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        # Project the initialization to the boundary.
        # log.info("before first binary_search_batch")
        # cur_iter = 0
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        original = images.cuda()
        perturbed = perturbed.cuda()
        starting_point = perturbed.cuda()


        patch_num = 7  # 横纵几等分
        patch_size = int(original.shape[2] / patch_num)

        success_num = 0  # 成功和失败的次数
        fail_num = 0

        value_mask = self.value_mask_init(patch_num)
        noise_mask = self.noise_mask_init(starting_point, original, patch_num, patch_size)

        best_noise = starting_point - original
        current_min_noise = self.l2_distance(starting_point, original)
        # init variables
        for j in range(10000):
            if torch.sum(value_mask * noise_mask) == 0:  # 当前平分方法下没有可以查询的了
                # FIXME
                print("patch num * 2", patch_num)
                patch_num *= 2

                if patch_num == 448:  # 没必要了
                    # print("only", j ,patch_num)
                    break

                patch_size = int(original.shape[2] / patch_num)
                # print( patch_size)
                value_mask = self.value_mask_init(patch_num)
                noise_mask = self.noise_mask_init(best_noise, original, patch_num, patch_size)

            total_mask = value_mask * noise_mask
            best_index = torch.argmax(total_mask)
            best_row, best_col = self.translate(best_index, patch_num)

            # print("best_row, best_col", best_row.item(), best_col.item())

            temp_noise = copy.deepcopy(best_noise)

            temp_noise[(best_row * patch_size):(best_row * patch_size + patch_size),
            (best_col * patch_size):(best_col * patch_size + patch_size)] = 0

            candidate = original + temp_noise

            if self.l2_distance(candidate, original) >= current_min_noise:
                # print("back")
                value_mask[best_row, best_col] = 0
                # print("not worth", torch.sum(value_mask).item())
                continue


            # #下面更新起点

            if self.decision_function(candidate,true_labels, target_labels):
                # print(self.l2_distance(candidate, original).item(), "Success")
                current_min_noise = self.l2_distance(candidate, original)
                success_num += 1
                query += 1
                best_noise = candidate - original
                noise_mask[best_row, best_col] = self.l2_distance(
                    best_noise[(best_row * patch_size):(best_row * patch_size + patch_size),
                    (best_col * patch_size):(best_col * patch_size + patch_size)], 0)
                perturbed = original + best_noise
                dist = torch.norm((perturbed - original).view(batch_size, -1), self.ord, 1)
                log.info('{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(batch_index + 1, j + 1,
                                                                                               self.norm, dist.item(),
                                                                                               int(query[0].item())))
            else:
                # print("Fail")
                fail_num += 1
                query += 1
                value_mask[best_row, best_col] = 0

            # cur_iter += 1
            if torch.sum(query >= self.maximum_queries).item() == true_labels.size(0):
                break
            # compute new distance.
            # dist = torch.norm((perturbed - original).view(batch_size, -1), self.ord, 1)
            # log.info('{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(batch_index+1, j + 1, self.norm, dist.item(), int(query[0].item())))
            if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                break

        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return perturbed, query, success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name, result_dump_path):
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

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()
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
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.float().cpu() * (success_query <= self.maximum_queries).float()

            for key in ['query', 'correct', 'not_done',
                        'success', 'success_query', "distortion_with_max_queries"]:
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
                          "success_all":self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


def get_exp_dir_name(dataset,  norm, targeted, target_type, random_direction, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.init_num_eval_grad!=100:
        if args.attack_defense:
            dirname = 'PAR@{}_on_defensive_model-{}-{}-{}'.format(args.init_num_eval_grad, dataset, norm, target_str)
        else:
            dirname = 'PAR@{}-{}-{}-{}'.format(args.init_num_eval_grad, dataset, norm, target_str)
    else:
        if random_direction:
            if args.attack_defense:
                dirname = 'PARRandom_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'PARRandom-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            if args.attack_defense:
                dirname = 'PAR_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'PAR-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/PAR.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type',type=str, default='increment', choices=['random', "load_random", 'least_likely',"increment"])
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument("--num_iterations",type=int,default=64)
    parser.add_argument('--stepsize_search', type=str, choices=['geometric_progression', 'grid_search'], default='geometric_progression')
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense_eps',type=str,default="")
    parser.add_argument('--random_direction',action="store_true")
    parser.add_argument('--max-queries',type=int, default=10000)
    parser.add_argument('--init-num-eval-grad', type=int, default=100)
    parser.add_argument('--gamma',type=float)

    args = parser.parse_args()
    assert args.batch_size == 1, "HSJA only supports mini-batch size equals 1!"
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
                            get_exp_dir_name(args.dataset, args.norm, args.targeted, args.target_type, args.random_direction, args))  # 随机产生一个目录用于实验
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
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
        attacker = PAR(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                     args.norm, args.load_random_class_image, args.epsilon,  args.num_iterations,
                                     max_num_evals=1e4, init_num_evals=args.init_num_eval_grad,
                                     maximum_queries=args.max_queries, random_direction=args.random_direction)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
