import argparse

import os
import random
import sys
sys.path.append(os.getcwd())
from collections import defaultdict, OrderedDict

import json
from types import SimpleNamespace
import os.path as osp
import glog as log
from torch.nn import functional as F
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from models.standard_model import StandardModel
from models.defensive_model import DefensiveModel

# from scipy.fftpack import dct, idct
import CGBA.torch_dct as torch_dct
import math

import numpy as np
import torch
import os
from utils.dataset_toolkit import select_random_image_of_target_class

class CGBA(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, norm, epsilon, load_random_class_image,
                 tol, sigma, init_rnd_adv, dim_reduce_factor, attack_method = 'CGBA_H',
                 iterations=93,initial_query=100,
                 grad_estimator_batch_size=40, maximum_queries=10000,batch_size=1
                 ):
        self.dim_reduce_factor = dim_reduce_factor
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
        self.init_rnd_adv = init_rnd_adv
        self.num_iterations = iterations
        self.N0 = initial_query

        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = grad_estimator_batch_size
        self.attack_method = attack_method

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)

    def decision_function(self, images, true_labels, target_labels):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1] != true_labels
        else:
            return logits.max(1)[1] == target_labels

    def is_adversarial(self, images):
        assert images.dim() == 4
        is_adv = self.decision_function(images, self.true_labels, self.target_labels)
        if is_adv:
            return 1
        else:
            return -1

    def initialize(self, sample, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        while True:
            random_noise = torch.from_numpy(
                np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float().to(sample.device)
            # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
            success = self.decision_function(random_noise[None], true_labels, target_labels)
            num_eval += 1
            if success:
                break
            if num_eval > 1000:
                log.info("Initialization failed! Use a misclassified image of a targeted class as the initial image.")
                if target_labels is None:
                    target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                  size=true_labels.size()).long().to(sample.device)
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0,
                                                                            high=CLASS_NUM[self.dataset_name],
                                                                            size=target_labels[
                                                                                invalid_target_index].size()).long().to(sample.device)
                        invalid_target_index = target_labels.eq(true_labels)

                initialization = select_random_image_of_target_class(self.dataset_name, target_labels, self.model,
                                                                     self.load_random_class_image).squeeze().to(sample.device)
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
        return initialization, num_eval

    def find_random_adversarial(self, image):
        num_calls = 1
        #step = 0.02
        step = 0.002        #--[debug]
        perturbed = image
        while self.is_adversarial(perturbed) == -1:
            pert = torch.randn(image.shape).to(image.device)
            perturbed = image + num_calls * step * pert
            perturbed = self.clip_image_values(perturbed)
            num_calls += 1
        return perturbed, num_calls

    def bin_search(self, x_0, x_random):
        num_calls = 0
        adv = x_random
        cln = x_0
        while True:
            mid = (cln + adv) / 2.0
            num_calls += 1
            if self.is_adversarial(mid) == 1:
                adv = mid
            else:
                cln = mid
            if torch.norm(adv-cln).cpu().numpy() < self.tol or num_calls >= 100:
                break
        return adv, num_calls

    def normal_vector_approximation_batch(self, x_boundary, q_max, random_noises):
        '''
        To estimate the normal vector on the boundary point, x_boundary, at each iteration
        '''
        grad_tmp = [] # estimated gradients in each estimate_batch
        z = [] # sign of grad_tmp
        outs = []
        num_batchs = math.ceil(q_max/self.grad_estimator_batch_size)
        last_batch = q_max - (num_batchs-1)*self.grad_estimator_batch_size
        num_calls = 0
        for j in range(num_batchs):
            if j == num_batchs-1:
                current_batch = random_noises[self.grad_estimator_batch_size * j:]
                # noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*last_batch + self.sigma*current_batch.cpu().numpy()
                #print(x_boundary.shape)
                #print(current_batch.shape)
                # if x_boundary.shape == torch.Size([self.channels, self.width, self.height]):
                #     x_boundary = x_boundary.unsqueeze(dim=0) #--[debug]
                noisy_boundary = x_boundary.repeat(last_batch, 1, 1, 1) + self.sigma * current_batch
            else:
                current_batch = random_noises[self.grad_estimator_batch_size * j:self.grad_estimator_batch_size * (j + 1)]
                noisy_boundary = x_boundary.repeat(self.grad_estimator_batch_size, 1, 1, 1) +self.sigma * current_batch
            # log.info("Gradient estimation costs {} queries".format(noisy_boundary.size(0)))
            predict_labels = torch.argmax(self.model.forward(noisy_boundary), 1) #.cpu().numpy().astype(int)
            num_calls += noisy_boundary.size(0)
            outs.append(predict_labels)
        outs = torch.cat(outs, dim=0)
        for i, predict_label in enumerate(outs):
            if self.target_labels == None:
                if predict_label == self.true_labels:
                    z.append(1)
                    grad_tmp.append(random_noises[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-random_noises[i])
            if self.target_labels != None:
                if predict_label != self.target_labels:
                    z.append(1)
                    grad_tmp.append(random_noises[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-random_noises[i])
        grad = -(1/q_max)*sum(grad_tmp)
        grad_f = grad.unsqueeze(0)
        return grad_f, sum(z), num_calls

    def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o/torch.norm(eta_o)
        v = (x_b - x_s)/torch.norm(x_b - x_s)
        theta = torch.acos(torch.dot(eta.reshape(-1), v.reshape(-1)))
        while True:
            #m = (torch.sin(theta.cpu())*torch.cos(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))/torch.sin(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))-torch.cos(theta.cpu())).item()
            m = (torch.sin(theta.cpu()) * torch.cos(
                torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) / torch.sin(
                torch.tensor([(math.pi / 2)]) * (1 - 1 / pow(2, num_calls))) - torch.cos(
                theta.cpu())).item()  # --[debug]

            zeta = (eta + m*v)/torch.norm(eta + m*v)
            p_near_boundary = x_s + zeta*torch.norm(x_b-x_s)*torch.dot(v.reshape(-1), zeta.reshape(-1))
            p_near_boundary = self.clip_image_values(p_near_boundary)
            num_calls += p_near_boundary.size(0)
            if self.is_adversarial(p_near_boundary) == -1 or num_calls > 100:
                if num_calls>100:
                    log.info("Finding initial boundary point failed!")
                break
        perturbed, n_calls = self.semi_circular_boundary_search(x_s, x_b, p_near_boundary)
        return perturbed, num_calls - 1 + n_calls

    def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o / torch.norm(eta_o).cuda()
        v = (x_b - x_s) / torch.norm(x_b - x_s)
        theta = torch.acos(torch.clamp(torch.dot(eta.reshape(-1), v.reshape(-1)), -1.0, 1.0))
        while True:
            # It may cause RuntimeError: Overflow when unpacking long, because 2^num_calls is too large for torch.sin and torch.cos.
            x = theta / float(pow(2, num_calls))
            m = (torch.sin(theta) * torch.cos(x) / torch.sin(x) - torch.cos(theta)).item()
            zeta = (eta + m * v) / torch.norm(eta + m * v)

            perturbed = x_s + zeta * torch.norm(x_b - x_s) * torch.dot(zeta.reshape(-1), v.reshape(-1))
            perturbed = self.clip_image_values(perturbed)
            num_calls += perturbed.size(0)
            if self.is_adversarial(perturbed) == 1 or num_calls > 100:
                if num_calls>100:
                    log.info("Finding initial boundary point failed!")
                break
        perturbed, bin_query = self.bin_search(x_s, perturbed)
        return perturbed, num_calls - 1 + bin_query

    def semi_circular_boundary_search(self, x_0, x_b, p_near_boundary):
        num_calls = 0
        norm_dis = torch.norm(x_b - x_0)
        boundary_dir = (x_b - x_0) / torch.norm(x_b - x_0)
        clean_dir = (p_near_boundary - x_0) / torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b
        clean = x_0
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir / torch.norm(mid_dir)
            # theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
            #             torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir)))
            theta = torch.acos(torch.clamp((torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
                    torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir))), -1.0, 1.0))
            d = torch.cos(theta) * norm_dis
            x_mid = x_0 + mid_dir * d
            num_calls += 1
            if self.is_adversarial(x_mid) == 1:
                adv_dir = mid_dir
                adv = x_mid
            else:
                clean_dir = mid_dir
                clean = x_mid
            if torch.norm(adv - clean).cpu().numpy() < self.tol:
                break
            if num_calls > 100:
                break
        return adv, num_calls

    # def ellipse_boundary_search(self, x_0, x_b, p_near_boundary):
    #     num_calls = 0
    #     norm_dis = torch.norm(x_b - x_0)
    #     boundary_dir = (x_b - x_0) / torch.norm(x_b - x_0)
    #
    #     # 定义椭圆的参数，根据实际情况调整
    #     semi_major_axis = torch.tensor([4.0]).to(x_0.device)
    #     semi_minor_axis = torch.tensor([1.0]).to(x_0.device)
    #
    #     clean_dir = (p_near_boundary - x_0) / torch.norm(p_near_boundary - x_0)
    #     adv_dir = boundary_dir
    #     adv = x_b
    #     clean = x_0
    #     while True:
    #         mid_dir = adv_dir + clean_dir
    #         mid_dir /= torch.norm(mid_dir)
    #
    #         theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) / (
    #                 torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir)))
    #         d = torch.cos(theta) * norm_dis
    #
    #         # 计算椭圆上的点位置
    #         a, b = semi_major_axis, semi_minor_axis
    #         aa = torch.dot(a, a)
    #         bb = torch.dot(b, b)
    #         mid_norm = torch.norm(mid_dir)
    #         x_mid = x_0 + mid_dir * d * aa / mid_norm
    #         y_mid = x_0 + mid_dir * d * bb / mid_norm
    #         norm_xy = torch.norm(torch.cat((x_mid.unsqueeze(0), y_mid.unsqueeze(0)), dim=0), dim=0)
    #         x_mid = x_0 + a * x_mid / norm_xy
    #         y_mid = x_0 + b * y_mid / norm_xy
    #
    #         num_calls += 1
    #         if self.is_adversarial(y_mid) == 1:
    #             adv_dir = mid_dir
    #             adv = y_mid
    #         else:
    #             clean_dir = mid_dir
    #             clean = y_mid
    #         if torch.norm(adv - clean).cpu().numpy() < self.tol:
    #             break
    #         if num_calls > 100:
    #             break
    #     return adv, num_calls

    def find_random(self, x, n):
        image_size = x.shape
        out = torch.zeros(n, 3, int(image_size[-2]), int(image_size[-1]))
        for i in range(n):
            random_x = torch.zeros_like(x)
            fill_size = int(image_size[-1] / self.dim_reduce_factor)
            random_x[:, :, :fill_size, :fill_size] = torch.randn(image_size[0], x.size(1), fill_size, fill_size)
            if self.dim_reduce_factor > 1.0:
                random_x = torch_dct.idct_2d(random_x, norm='ortho')
                # random_x = torch.from_numpy(
                #     idct(idct(random_x.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
            out[i] = random_x
        return out


    def clip_image_values(self, x):
        x = torch.clamp(x, self.clip_min, self.clip_max)
        return x

    def attack(self, batch_index, images, target_images, true_labels, target_labels):
        query = torch.zeros_like(true_labels).float()
        success_stop_queries = query.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1)*self.batch_size, self.total_images)).tolist()
        assert images.size(0) == 1
        batch_size = images.size(0)
        images = images.cuda()
        self.true_labels = true_labels.to(images.device)
        if target_labels is not None:
            target_images = target_images.to(images.device)
            self.target_labels = target_labels.to(images.device)
        else:
            self.target_labels = target_labels

        # Initialize. Note that if the original image is already classified incorrectly, the difference between the found initialization and sample is very very small, this case will lead to inifinity loop later.
        if self.init_rnd_adv:
            if target_labels is None:
                x_adv, query_random = self.find_random_adversarial(images)
            else:
                x_adv, query_random = target_images, 0
            x_adv, num_eval = self.bin_search(images, x_adv)
            query += num_eval + query_random
        else:
            if self.target_labels is None:
                x_adv, query_random = self.initialize(images, self.true_labels, self.target_labels)
                query += query_random
            else:
                x_adv, query_random = target_images, 0
                x_adv, num_eval = self.bin_search(images, x_adv)
                query += num_eval + query_random
        dist = torch.norm((x_adv - images).view(batch_size, -1), self.ord,1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1).detach().cpu()
        success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

        size = images.shape
        # init variables
        for j in range(self.num_iterations):
            q_opt = int(self.N0 * np.sqrt(j+1))
            # search step size.
            if self.dim_reduce_factor < 1.0:
                raise Exception("The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
            if self.dim_reduce_factor > 1.0:
                random_vec_o = self.find_random(images, q_opt).to(images.device)
            else:
                # print('The attack is performing in full-dimensional image space')
                random_vec_o = torch.randn(q_opt, self.channels, size[-2],size[-1]).to(images.device)

            grad_oi, ratios, calls = self.normal_vector_approximation_batch(x_adv, q_opt, random_vec_o)
            query += calls
            if self.attack_method == 'CGBA':
                x_adv, qs = self.go_to_boundary_CGBA(images, grad_oi, x_adv)
            if self.attack_method == 'CGBA_H':
                x_adv, qs = self.go_to_boundary_CGBA_H(images, grad_oi, x_adv)
            query += qs
            dist = torch.norm((x_adv - images).view(batch_size, -1), self.ord, 1)
            working_ind = torch.nonzero(dist > self.epsilon).view(-1).detach().cpu()
            success_stop_queries[working_ind] = query[working_ind]
            for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[
                    inside_batch_index].item()

            log.info('{}-th image, iteration: {}, {}: distortion {:.4f}, query: {}'.format(batch_index+1, j + 1, self.norm, dist.item(), int(query[0].item())))
            if dist.item() < 1e-4:
                break
            if query >= self.maximum_queries:
                break
        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return x_adv, query, success_stop_queries, dist, (dist <= self.epsilon)


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
                adv_logit = self.model(adv_images)
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
        dirname = '{}_on_defensive_model-{}-{}-{}'.format(args.attack_method, dataset, norm, target_str)
    else:
        dirname = '{}-{}-{}-{}'.format(args.attack_method, dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/CGBA.json',
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
    parser.add_argument('--init-rnd-adv', action='store_true')
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack_defense',action="store_true")
    parser.add_argument("--num_iterations",type=int,default=200)
    parser.add_argument("--initial-query", type=int, default=100)
    parser.add_argument('--defense_model',type=str, default=None)
    parser.add_argument('--defense_norm',type=str,choices=["l2","linf"],default='linf')
    parser.add_argument('--defense_eps',type=str,default="")
    parser.add_argument('--max_queries',type=int, default=10000)
    parser.add_argument('--attack_method', type=str,choices=['CGBA_H',"CGBA"], required=True)
    parser.add_argument('--dim_reduce_factor', type=int)

    args = parser.parse_args()
    assert args.batch_size == 1
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
        # args.max_queries = 20000
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
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
        attacker = CGBA(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1], IN_CHANNELS[args.dataset],
                                     args.norm, args.epsilon, args.load_random_class_image,args.tol,args.sigma,args.init_rnd_adv,
                                    args.dim_reduce_factor,args.attack_method, args.num_iterations,args.initial_query,
                                    grad_estimator_batch_size=40,
                                     maximum_queries=args.max_queries)
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()