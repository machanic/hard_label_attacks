# -*- coding: gbk -*-
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
import torch.nn as nn
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel
from utils.dataset_toolkit import select_random_image_of_target_class


class Configuration:
    def __init__(self, on=False, name="none", eps=0, eta=0, alp=0, c=0, lr=0, iter=0, sigma=0, stop=False):
        self.on = on
        self.name = name
        self.eps = eps
        self.eta = eta
        self.c = c
        self.alp = alp
        self.lr = lr
        self.iter = iter  # steps
        self.sigma = sigma
        self.stop = stop
        return

class SQBA_Attack(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, norm, epsilon, load_random_class_image,
                 surrogate_model,  dgm_cfg=None, sqba_cfg=None,
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
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.height = height
        self.width = width
        self.channels = channels
        # self.shape = (channels, height, width)
        self.load_random_class_image = load_random_class_image
        self.surrogate_model = surrogate_model
        self.surrogate_model.loss = 'cross entropy'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.targeted = False

        self.dgm_cfg = dgm_cfg
        self.sqba_cfg = sqba_cfg

        # self.total_images = total_images
        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images) # query times
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).to(self.device)
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
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float().to(self.device)
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise, true_labels, target_labels)
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long().to(self.device)
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                                size=target_labels[invalid_target_index].size()).long().to(self.device)
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = select_random_image_of_target_class(self.dataset_name, target_labels, self.model, self.load_random_class_image).squeeze().to(self.device)
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

    def check_adversary(self, x):
        _, pred = self.model(x.reshape(self.shape)).data.max(1)
        self.query += 1
        flag = False
        if self.targeted:
            if pred == self.label:
                flag = True
        else:
            if pred != self.label:
                flag = True
        return flag

    def lossf_list(self, type='cross entropy'):

        if type == 'cross entropy':
            lossF = nn.CrossEntropyLoss()

        return lossF

    def setup(self, x):
        self.shape = x.size()
        x_ = x.detach().clone().flatten()
        self.x = x_

        self.d_prev = np.inf

        # Set binary search threshold
        if self.norm == 'l2':
            self.theta = 0.01 / np.sqrt(np.prod(self.shape))
        else:
            self.theta = 0.01 / np.prod(self.shape)
        # if self.norm == 'l2':
        #     self.theta = 1 / np.sqrt(np.prod(self.shape) * np.prod(self.shape))
        # else:
        #     self.theta = 1 / (np.prod(self.shape) ** 2)


    def _interpolate(self, current_sample, alpha):

        if self.norm == 'l2':
            result = (1 - alpha) * self.x + alpha * current_sample
        else:
            lb = torch.min((self.x - alpha).flatten())
            hb = torch.max((self.x + alpha).flatten())
            result = torch.clamp(current_sample, lb, hb)

        return result


    def _compute_delta(self, current_sample):
        if self.iter_cnt == 0:
            return 0.1 * (self.clip_max - self.clip_min)

        if self.norm == 'l2':
            dist = torch.norm(self.x - current_sample)
            delta = np.sqrt(np.prod(current_sample.size())) * self.theta * dist
        else:
            dist = torch.max(torch.abs(self.x - current_sample))
            delta = np.prod(current_sample.size()) * self.theta * dist
            delta = delta + 1e-17

        return delta

    def _binary_search(self, current_sample, threshold):
        # First set upper and lower bounds as well as the threshold for the binary search
        if self.norm == 'l2':
            upper_bound = torch.tensor(1, dtype=torch.float)
            lower_bound = torch.tensor(0, dtype=torch.float)

            if threshold is None:
                threshold = self.theta

        else:
            upper_bound = torch.max(torch.abs(self.x - current_sample))
            upper_bound = upper_bound.cpu()
            lower_bound = torch.tensor(0, dtype=torch.float)

            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample, alpha)

            # Update upper_bound and lower_bound
            satisfied = self.check_adversary(interpolated_sample)

            lower_bound = torch.where(torch.tensor(satisfied == 0), alpha, lower_bound)
            upper_bound = torch.where(torch.tensor(satisfied == 1), alpha, upper_bound)

            if self.query >= self.maximum_queries:
                break

        result = self._interpolate(current_sample, upper_bound)

        return result


    def dgm(self, x, yn, yp):
        x.requires_grad = True

        output = self.surrogate_model(x)

        loss = self.lossF(output, yn)
        gn = torch.autograd.grad(loss, x, retain_graph=True, create_graph=False)[0]
        loss = self.lossF(output, yp)
        gp = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

        x.requires_grad = False

        eps = 0.8
        gn = gn.sign() * (1)
        gp = gp.sign() * (-1)

        perturb = (eps * gn) + ((1 - eps) * gp)

        return perturb

    def dgm_wrap(self, x):
        output = self.surrogate_model(x)

        # choose positive direction with the highest probability class
        _, i_p = torch.sort(output, descending=True)
        for i in range(i_p.size()[1]):
            if i_p[0, i] != self.label.reshape([1]):
                yp = i_p[0, i].reshape(1)
                break
        g = self.dgm(x, self.label.reshape([1]), yp)

        return g

    def make_sample(self, x_adv):
        x_adv = x_adv.reshape(self.shape)
        g = self.dgm_wrap(x_adv)
        g = g.flatten()

        return g

    def white_grad(self, x_adv, delta):
        v = x_adv - self.x

        delta = delta*2
        num = 2
        bias = 0.1
        a = (1 - (bias * 2)) / (num - 1)
        r = np.zeros(num)
        for i in range(num):
            r[i] = bias + (a * i)

        d_list = torch.zeros([1, num])
        a_list = torch.zeros([1, num])
        g_list = []
        x_list = []
        for i in range(num):
            g_t = self.make_sample(self.x + v * r[i])

            aa = g_t.flatten()
            bb = v.flatten()
            cc = torch.matmul(aa, bb)
            angle = cc / (torch.norm(aa) * torch.norm(bb))

            x_t = torch.clamp((x_adv + (delta * g_t)), self.clip_min, self.clip_max)
            d_t = torch.norm(self.x - x_t)

            x_list.append(x_t)
            g_list.append(g_t)
            d_list[0, i] = d_t
            a_list[0, i] = angle

        d_sort, indices = torch.sort(d_list)
        for i in range(num):
            idx = indices[0, i]
            if self.check_adversary(x_list[idx]):
                break
        else:
            idx = indices[0, 0]
            print("failed adversarial")

        return g_list[idx]

    def white_update(self, x_adv, delta):
        u_t = self.white_grad(x_adv, delta)

        x_t = torch.clamp((x_adv + delta * u_t), self.clip_min, self.clip_max)
        dh = (x_t - x_adv) / delta
        g = dh / torch.norm(dh)

        return g

    def black_update(self, x_adv, delta, num_eval):
        x_adv = x_adv.clone().flatten()

        if delta == 0 or num_eval == 0:
            print("delta {:.3f}, num_eval {}".format(delta, num_eval))

        rnd_noise_shape = [num_eval] + list(x_adv.size())
        if self.norm == 'l2':
            rnd_noise = torch.randn(rnd_noise_shape).to(self.device)
        else:
            rnd_noise = torch.rand(rnd_noise_shape).to(self.device)

        # Normalize random noise to fit into the range of input data
        rnd_noise = rnd_noise / torch.sqrt(
            torch.sum(rnd_noise ** 2, dim=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True))

        eval_samples = torch.clamp(x_adv + delta * rnd_noise, self.clip_min, self.clip_max)
        rnd_noise = (eval_samples - x_adv) / delta

        satisfied = torch.zeros(num_eval)
        for i in range(num_eval):
            satisfied[i] = self.check_adversary(eval_samples[i, :].reshape(self.shape))

        f_val = 2 * satisfied - 1.0
        f_val = f_val.to(self.device)

        if torch.mean(f_val) == 1.0:
            grad = torch.mean(rnd_noise, dim=0)
        elif torch.mean(f_val) == -1.0:
            grad = -torch.mean(rnd_noise, dim=0)
        else:
            m = torch.mean(f_val)
            f_val -= m
            f_val = f_val.reshape([len(f_val), 1])
            grad = torch.mean(f_val * rnd_noise, dim=0)

        # Compute update
        if self.norm == 'l2':
            g = grad / torch.norm(grad)
        else:
            g = torch.sign(grad)

        if torch.isnan(g).any():
            print("result Nan - {:.5f}".format(torch.norm(grad)))

        return g

    def compute_update(self, x_adv, delta):
        if self.white_only:
            g = self.white_update(x_adv, delta)
        else:
            num_eval = int(self.min_randoms * np.sqrt(self.iter_cnt + 1))
            num_eval = min(int(num_eval), np.abs(self.maximum_queries - self.query.item()))
            g = self.black_update(x_adv, delta, num_eval)

        return g

    def perturb(self, x_best):
        # intermediate adversarial example
        x_prev = x_best.clone()
        x_adv = x_best.clone()

        success_cnt = 0

        delta = self._compute_delta(x_adv)

        # Then run binary search
        x_adv = self._binary_search(x_adv, self.threshold)
        if self.query >= self.maximum_queries:
            return x_adv
        # Next compute the number of evaluations and compute the update
        update = self.compute_update(x_adv, delta)

        # Finally run step size search by first computing epsilon
        if self.norm == 'l2':
            dist = torch.norm(self.x - x_adv)
        else:
            dist = torch.max(torch.abs(self.x - x_adv))

        epsilon = 2.0 * dist / np.sqrt(self.iter_cnt + 1)

        while True:
            epsilon /= 2.0
            x_c = x_adv + epsilon * update
            success = self.check_adversary(torch.clamp(x_c, self.clip_min, self.clip_max))
            if success:
                break

        if self.white_only:
            if dist < self.d_prev:
                self.d_prev = dist
                x_best = x_adv
        else:
            x_best = x_adv

        if epsilon < 1.0:
            # stop using updates from surrogate model
            self.white_only = False
            x_best = x_adv

        # Update current sample
        if success is True:
            x_best = torch.clamp(x_c, self.clip_min, self.clip_max)
            success_cnt += 1

        #print("{} - {:.4f} {:.4f} {:.4f}".format(self.iter_cnt, dist, epsilon, delta))
        # Update current iteration
        self.iter_cnt += 1

        if success_cnt == 0:
            print('failed - converge')

        if torch.isnan(x_best).any():  # pragma: no cover
            x_best = x_prev

        return x_best

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        self.query = torch.zeros_like(true_labels)#.float()
        success_stop_queries = self.query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()
        batch_size = images.size(0)

        x = images.to(self.device)
        self.label = true_labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)

        self.threshold = 0.001
        self.iter_cnt = 0
        self.min_randoms = 10
        self.white_only = True

        self.lossF = self.lossf_list(self.surrogate_model.loss).to(self.device)
        if self.sqba_cfg.on:
            self.setup(x)
            sm = 0
            while self.query < self.maximum_queries:
                if sm == 0:
                    x_adv, num_eval = self.initialize(x, target_images, self.label, target_labels)
                    x_adv = x_adv.flatten()

                    self.query += num_eval
                    sm = 1
                    dist = torch.norm((x_adv - self.x).view(batch_size, -1), self.ord, 1).cpu()
                    best_adv, best_dist = x_adv, dist

                    working_ind = torch.nonzero(dist > self.epsilon).view(
                        -1)  # get locations (i.e., indexes) of non-zero elements of an array.
                    success_stop_queries[working_ind] = self.query[working_ind]  # success times

                    for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                        self.distortion_all[index_over_all_images][self.query[inside_batch_index].item()] = dist[
                            inside_batch_index].item()

                elif sm == 1:
                    x_adv = self.perturb(x_adv)

                    dist = torch.norm((x_adv - self.x).view(batch_size, -1), self.ord, 1).cpu()
                    if dist < best_dist:
                        best_adv, best_dist = x_adv, dist

                    working_ind = torch.nonzero(dist > self.epsilon).view(-1)
                    success_stop_queries[working_ind] = self.query[working_ind]
                    for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                        self.distortion_all[index_over_all_images][self.query[inside_batch_index].item()] = dist[
                            inside_batch_index].item()
                    log.info(
                        '{}-th image, {}: distortion {:.4f}, query: {}'.format(batch_index + 1, self.norm, dist.item(),
                                                                               int(self.query[0].item())))

                if dist.item() < 1e-4:  # 发现攻击jpeg时候卡住，故意加上这句话
                    break

        x_adv = best_adv.reshape(self.shape)
        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return x_adv, self.query, success_stop_queries, dist, (dist <= self.epsilon)

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
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.detach().cpu() * (
                        success_query.detach().cpu() <= self.maximum_queries).float()

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


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'SQBA_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        dirname = 'SQBA-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/SQBA.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1, help='batch size must set to 1')
    parser.add_argument('--dataset', type=str, required=True,
               choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"], help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all-archs', action="store_true")
    parser.add_argument("--surrogate-arch", type=str, required=True, help="the architecture of a surrogate model")
    parser.add_argument("--surrogate-defense-model", type=str, help="multiple surrogate models, and this parameter should be passed in through space splitting")
    parser.add_argument('--surrogate-defense-norm', type=str, choices=["l2", "linf"],
                        help="defense norms of multiple surrogate defense models, e.g., l2, linf")
    parser.add_argument('--surrogate-defense-eps', type=str, choices=["8_div_255", "4_div_255", "3"],
                        help="defense epsilon of multiple surrogate defense models, e.g., 3,4_div_255,8_div_255")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type', type=str, default='increment', choices=['random', "load_random", 'least_likely',"increment"])
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack-defense',action="store_true")
    parser.add_argument('--defense-model',type=str, default=None)
    parser.add_argument('--defense-norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense-eps',type=str,default="")
    parser.add_argument('--max-queries',type=int, default=10000)

    args = parser.parse_args()
    assert args.batch_size == 1, "SQBA only supports mini-batch size equals 1!"
    assert not args.targeted, "targeted attack is not supported yet"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    if args.targeted and args.dataset == "ImageNet":
        args.max_queries = 20000
    if args.attack_defense and args.defense_model == "adv_train_on_ImageNet":
        args.max_queries = 20000
    if not args.surrogate_defense_norm:
        args.surrogate_defense_norm = args.defense_norm
    if not args.surrogate_defense_eps:
        args.surrogate_defense_eps = args.defense_eps

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
            if args.surrogate_defense_model.startswith(
                    "adv_train") or args.surrogate_defense_model == "TRADES" or args.surrogate_defense_model == "feature_scatter":
                surrogate_str = "{surrogate_arch}({surrogate_defense_model}_{surrogate_defense_norm}_{surrogate_defense_eps})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model="AT" if args.surrogate_defense_model.startswith(
                        "adv_train") else args.surrogate_defense_model,
                    surrogate_defense_norm=args.surrogate_defense_norm,
                    surrogate_defense_eps=args.surrogate_defense_eps)
            else:
                surrogate_str = "{surrogate_arch}({surrogate_defense_model})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model="AT" if args.surrogate_defense_model.startswith(
                        "adv_train") else args.surrogate_defense_model)
            if args.defense_model.startswith("adv_train") or args.defense_model == "TRADES" or args.defense_model == "feature_scatter":
                log_file_path = osp.join(args.exp_dir, "run_{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps})_surrogate_{surrogate_arch}.log".format(
                    arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith("adv_train") else args.defense_model,
                    arch_defense_norm=args.defense_norm, arch_defense_eps=args.defense_eps, surrogate_arch=surrogate_str))
            else:
                log_file_path = osp.join(args.exp_dir, "run_{arch}({arch_defense_model})_surrogate_{surrogate_arch}.log".format(
                    arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith("adv_train") else args.defense_model,
                    surrogate_arch=surrogate_str))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}_surrogate_{}.log'.format(args.arch, args.surrogate_arch))
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
    for arch in archs:
        if args.attack_defense:
            if args.surrogate_defense_model.startswith("adv_train") or args.surrogate_defense_model == "TRADES" or args.surrogate_defense_model == "feature_scatter":
                surrogate_str = "{surrogate_arch}({surrogate_defense_model}_{surrogate_defense_norm}_{surrogate_defense_eps})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model="AT" if args.surrogate_defense_model.startswith(
                        "adv_train") else args.surrogate_defense_model,
                    surrogate_defense_norm=args.surrogate_defense_norm, surrogate_defense_eps=args.surrogate_defense_eps)
            else:
                surrogate_str = "{surrogate_arch}({surrogate_defense_model})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model="AT" if args.surrogate_defense_model.startswith(
                        "adv_train") else args.surrogate_defense_model)
            if args.defense_model.startswith("adv_train") or args.defense_model == "TRADES" or args.defense_model == "feature_scatter":
                save_result_path = args.exp_dir + "/{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps})_surrogate_{surrogate_archs}.json".format(
                    arch=args.arch,
                    arch_defense_model="AT" if args.defense_model.startswith("adv_train") else args.defense_model,
                    arch_defense_norm=args.defense_norm,
                    arch_defense_eps=args.defense_eps, surrogate_archs=surrogate_str)
            else:
                save_result_path = args.exp_dir + "/{arch}({arch_defense_model})_surrogate_{surrogate_archs}.json".format(
                    arch=args.arch,
                    arch_defense_model="AT" if args.defense_model.startswith("adv_train") else args.defense_model,
                    surrogate_archs=surrogate_str)
        else:
            save_result_path = args.exp_dir + "/{arch}_surrogate_{surrogate_arch}.json".format(arch=arch,
                                                                                  surrogate_arch=args.surrogate_arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,
                                   norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)

        surrogate_model = DefensiveModel(args.dataset, args.surrogate_arch, no_grad=False,
                                         defense_model=args.surrogate_defense_model, norm=args.surrogate_defense_norm,
                                         eps=args.surrogate_defense_eps)
        surrogate_model.cuda()
        surrogate_model.eval()
        # dgm_cfg = Configuration(False, "dgm", eps=0.005, alp=0.001, iter=300, c=0.3)
        sqba_cfg = Configuration(True, "sqba", iter=10000)

        model.cuda()
        model.eval()
        attacker = SQBA_Attack(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                args.norm, args.epsilon, args.load_random_class_image, surrogate_model,
                                sqba_cfg=sqba_cfg,  maximum_queries=args.max_queries, batch_size=args.batch_size)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
