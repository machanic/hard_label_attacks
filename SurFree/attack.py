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
import math
from scipy import fft

def atleast_kdim(x, ndim):
    shape = x.shape + (1,) * (ndim - len(x.shape))
    return x.reshape(shape)

class SurFree(object):
    def __init__(self, model, dataset, clip_min, clip_max, height, width, channels, batch_size, epsilon, norm,
                 load_random_class_image, maximum_queries=10000, BS_gamma: float = 0.01, theta_max: float = 30,
                 BS_max_iteration: int = 10, n_ortho: int = 100, rho: float = 0.98, T: int = 3,
                 with_alpha_line_search: bool = True, with_distance_line_search: bool = False,
                 with_interpolation: bool = False):
        self.model = model
        self.batch_size = batch_size
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, model.arch)
        self.total_images = len(self.dataset_loader.dataset)
        self.norm = norm
        self.ord = np.inf if self.norm == "linf" else 2
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.shape = (channels, height, width)
        self.load_random_class_image = load_random_class_image

        # Attack Parameters
        self._BS_gamma = BS_gamma
        self._theta_max = theta_max
        self._BS_max_iteration = BS_max_iteration
        self.T = T
        self.rho = rho
        assert self.rho <= 1 and self.rho > 0
        self.n_ortho = n_ortho
        self._directions_ortho = {}
        self._nqueries = []

        # Add or remove some parts of the attack
        self.with_alpha_line_search = with_alpha_line_search
        self.with_distance_line_search = with_distance_line_search
        self.with_interpolation = with_interpolation
        if self.with_interpolation and not self.with_distance_line_search:
            Warning("It's higly recommended to use Interpolation with distance line search.")

        self.maximum_queries = maximum_queries
        self.dataset_name = dataset

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
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max)
        logits = self.model(images)
        if target_labels is None:
            return logits.max(1)[1] != true_labels
        else:
            return logits.max(1)[1] == target_labels

    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(
                    np.random.uniform(self.clip_min, self.clip_max, size=self.shape)).float().to(sample.device)
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.decision_function(random_noise[None], true_labels, target_labels)
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                      size=true_labels.size()).long().to(sample.device)
                        invalid_target_index = target_labels.eq(true_labels)
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset_name],
                                                                size=target_labels[invalid_target_index].size()).long().to(sample.device)
                            invalid_target_index = target_labels.eq(true_labels)

                    initialization = select_random_image_of_target_class(self.dataset_name, [target_labels], self.model,
                                                                         self.load_random_class_image).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise
                success = self.decision_function(blended[None], true_labels, target_labels)
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

    def batch_initialize(self, samples, target_images, true_labels, target_labels):
        initialization = samples.clone()
        num_evals = torch.zeros_like(true_labels).float()

        with torch.no_grad():
            logit = self.model(samples)
        pred = logit.argmax(dim=1)
        correct = pred.eq(true_labels).float()
        for i in range(len(correct)):
            if target_images is None:
                initialization[i], num_evals[i] = self.initialize(samples[i], None, true_labels[i], None)
            else:
                initialization[i], num_evals[i] = self.initialize(samples[i], target_images[i], true_labels[i], target_labels[i])

        return initialization, num_evals

    def _is_adversarial(self, perturbed):
        # Count the queries made for each image
        # Count if the vector is different from the null vector
        for i, p in enumerate(perturbed):
            if not (p == 0).all():
                self._nqueries[i] += 1

        if self.target_labels is not None:
            return self.model(perturbed).argmax(1) == self.target_labels
        else:
            return self.model(perturbed).argmax(1) != self.true_labels

    def _get_candidates(self, originals, best_advs):
        """
        Find the lowest epsilon to misclassified x following the direction: q of class 1 / q + eps*direction of class 0
        """
        epsilons = torch.zeros(len(originals)).to(originals.device)
        direction_2 = torch.zeros_like(originals)
        while (epsilons == 0).any():
            direction_2 = torch.where(
                atleast_kdim(epsilons == 0, direction_2.ndim),
                self._basis.get_vector(self._directions_ortho),
                direction_2
            )

            for i, eps_i in enumerate(epsilons):
                if eps_i == 0:
                    # Concatenate the first directions and the last directions generated
                    self._directions_ortho[i] = torch.cat((
                        self._directions_ortho[i][:1],
                        self._directions_ortho[i][1 + len(self._directions_ortho[i]) - self.n_ortho:],
                        direction_2[i].unsqueeze(0)), dim=0)

            function_evolution = self._get_evolution_function(originals, best_advs, direction_2)
            new_epsilons = self._get_best_theta(originals, function_evolution, epsilons)

            self.theta_max = torch.where(new_epsilons == 0, self.theta_max * self.rho, self.theta_max)
            self.theta_max = torch.where((new_epsilons != 0) * (epsilons == 0), self.theta_max / self.rho, self.theta_max)
            epsilons = new_epsilons

        epsilons = epsilons.unsqueeze(0)
        if self.with_interpolation:
            epsilons = torch.cat((epsilons, epsilons[0] / 2), dim=0)

        candidates = torch.cat([function_evolution(eps).unsqueeze(0) for eps in epsilons], dim=0)

        if self.with_interpolation:
            d = self.distance(best_advs, originals)
            delta = self.distance(self._binary_search(originals, candidates[1], boost=True), originals)
            theta_star = epsilons[0]

            num = theta_star * (4 * delta - d * (torch.cos(theta_star) + 3))
            den = 4 * (2 * delta - d * (torch.cos(theta_star) + 1))

            theta_hat = num / den
            q_interp = function_evolution(theta_hat)
            if self.with_distance_line_search:
                q_interp = self._binary_search(originals, q_interp, boost=True)
            candidates = torch.cat((candidates, q_interp.unqueeze(0)), dim=0)

        return candidates

    def _get_evolution_function(self, originals, best_advs, direction_2):
        distances = self.distance(best_advs, originals)
        direction_1 = (best_advs - originals).flatten(1) / distances.reshape((-1, 1))
        direction_1 = direction_1.reshape(originals.shape)
        return lambda theta: (
                    originals + self._add_step_in_circular_direction(direction_1, direction_2, distances, theta)).clip(
            0, 1)

    def _get_best_theta(self, originals, function_evolution, best_params):
        coefficients = torch.zeros(2 * self.T)
        for i in range(0, self.T):
            coefficients[2 * i] = 1 - (i / self.T)
            coefficients[2 * i + 1] = - coefficients[2 * i]

        for i, coeff in enumerate(coefficients):
            params = coeff * self.theta_max
            x_evol = function_evolution(params)
            x = torch.where(atleast_kdim(best_params == 0, len(originals.shape)), x_evol, torch.zeros_like(originals))
            is_advs = self._is_adversarial(x)
            best_params = torch.where(
                torch.logical_and(best_params == 0, is_advs),
                params,
                best_params
            )
        if (best_params == 0).all() or not self.with_alpha_line_search:
            return best_params
        else:
            return self._alpha_binary_search(function_evolution, best_params, best_params != 0)

    def _alpha_binary_search(self, function_evolution, lower, mask):
        # Upper --> not adversarial /  Lower --> adversarial

        def get_alpha(theta):
            return 1 - torch.cos(theta * np.pi / 180)

        check_opposite = lower > 0  # if param < 0: abs(param) doesn't work

        # Get the upper range
        upper = torch.where(
            torch.logical_and(abs(lower) != self.theta_max, mask),
            lower + torch.sign(lower) * self.theta_max / self.T,
            torch.zeros_like(lower)
        )

        mask_upper = torch.logical_and(upper == 0, mask)
        while mask_upper.any():
            # Find the correct lower/upper range
            upper = torch.where(
                mask_upper,
                lower + torch.sign(lower) * self.theta_max / self.T,
                upper
            )
            x = function_evolution(upper)

            mask_upper = mask_upper * self._is_adversarial(x) * (self._nqueries < self.maximum_queries)
            lower = torch.where(mask_upper, upper, lower)

        step = 0
        while step < self._BS_max_iteration and (abs(get_alpha(upper) - get_alpha(lower)) > self._BS_gamma).any():
            mid_bound = (upper + lower) / 2
            mid = function_evolution(mid_bound)
            is_adv = self._is_adversarial(mid)

            mid_opp = torch.where(
                atleast_kdim(check_opposite, mid.ndim),
                function_evolution(-mid_bound),
                torch.zeros_like(mid)
            )
            is_adv_opp = self._is_adversarial(mid_opp)

            lower = torch.where(mask * is_adv, mid_bound, lower)
            lower = torch.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, -mid_bound, lower)
            upper = torch.where(mask * is_adv.logical_not() * check_opposite * is_adv_opp, - upper, upper)
            upper = torch.where(mask * (abs(lower) != abs(mid_bound)), mid_bound, upper)

            check_opposite = mask * check_opposite * is_adv_opp * (lower > 0)

            step += 1
        return lower

    def _binary_search(self, originals, perturbed, boost=False):
        # Choose upper thresholds in binary search based on constraint.
        highs = torch.ones(len(perturbed)).to(perturbed.device)
        d = np.prod(perturbed.shape[1:])
        thresholds = self._BS_gamma / (d * math.sqrt(d))
        lows = torch.zeros_like(highs)

        # Boost Binary search
        if boost:
            boost_vec = 0.1 * originals + 0.9 * perturbed
            is_advs = self._is_adversarial(boost_vec)
            is_advs = atleast_kdim(is_advs, originals.ndim)
            originals = torch.where(is_advs.logical_not(), boost_vec, originals)
            perturbed = torch.where(is_advs, boost_vec, perturbed)

        # use this variable to check when mids stays constant and the BS has converged
        old_mids = highs
        iteration = 0
        while torch.any(highs - lows > thresholds) and iteration < self._BS_max_iteration:
            iteration += 1
            mids = (lows + highs) / 2
            mids_perturbed = self._project(originals, perturbed, mids)
            is_adversarial_ = self._is_adversarial(mids_perturbed)

            highs = torch.where(is_adversarial_, mids, highs)
            lows = torch.where(is_adversarial_, lows, mids)

            # check of there is no more progress due to numerical imprecision
            reached_numerical_precision = (old_mids == mids).all()
            old_mids = mids
            if reached_numerical_precision:
                break

        results = self._project(originals, perturbed, highs)
        return results

    def _project(self, originals, perturbed, epsilons):
        epsilons = atleast_kdim(epsilons, originals.ndim)
        return (1.0 - epsilons) * originals + epsilons * perturbed

    def _add_step_in_circular_direction(self, direction1, direction2, r, degree):
        degree = atleast_kdim(degree, len(direction1.shape))
        r = atleast_kdim(r, len(direction1.shape))
        results = torch.cos(degree * np.pi / 180) * direction1 + torch.sin(degree * np.pi / 180) * direction2
        results = results * r * torch.cos(degree * np.pi / 180)
        return results

    def distance(self, a, b):
        return (a - b).flatten(1).norm(dim=1)

    def attack(self, batch_index, images, target_images, true_labels, target_labels, **kwargs):
        images = images.cuda()
        batch_size = images.size(0)
        self.true_labels = true_labels.cuda()
        self.target_images = target_images if target_labels is None else target_images.cuda()
        self.target_labels = target_labels if target_labels is None else target_labels.cuda()

        self._nqueries = torch.zeros(len(images)).to(images.device)#{i: 0 for i in range(len(originals))}
        self.theta_max = torch.ones(len(images)).to(images.device) * self._theta_max

        success_stop_queries = self._nqueries.clone()  # stop query count once the distortion < epsilon
        batch_image_positions = np.arange(batch_index * self.batch_size,
                                          min((batch_index + 1) * self.batch_size, self.total_images)).tolist()

        # Get Starting Point
        best_advs, num_evals = self.batch_initialize(images, self.target_images, self.true_labels, self.target_labels)
        self._nqueries += num_evals
        # best_advs = self._binary_search(images, best_advs, boost=True)

        dist = torch.norm((best_advs - images).view(batch_size, -1), self.ord, 1)
        working_ind = torch.nonzero(dist > self.epsilon).view(-1)
        success_stop_queries[working_ind] = self._nqueries[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][self._nqueries[inside_batch_index].item()] = dist[
                inside_batch_index].item()

        # assert self._is_adversarial(best_advs).all()
        # Initialize the direction orthogonalized with the first direction
        fd = best_advs - images
        norm = torch.norm(fd.view(batch_size, -1), self.ord, 1)
        fd = fd / atleast_kdim(norm, len(fd.shape))
        self._directions_ortho = {i: v.unsqueeze(0) for i, v in enumerate(fd)}

        # Load Basis
        self._basis = Basis(images, **kwargs["basis_params"]) if "basis_params" in kwargs else Basis(images)
        while not all(v > self.maximum_queries for v in self._nqueries):
            # Get candidates. Shape: (n_candidates, batch_size, image_size)
            candidates = self._get_candidates(images, best_advs)
            candidates = candidates.transpose(1, 0)

            best_candidates = torch.zeros_like(best_advs)
            for i, o in enumerate(images):
                o_repeated = torch.cat([o.unsqueeze(0)] * len(candidates[i]), dim=0)
                index = self.distance(o_repeated, candidates[i]).argmax()
                best_candidates[i] = candidates[i][index]

            is_success = self.distance(best_candidates, images) < self.distance(best_advs, images)
            best_advs = torch.where(atleast_kdim(is_success, best_candidates.ndim), best_candidates, best_advs)

            dist = torch.norm((best_advs - images).view(batch_size, -1), self.ord, 1)
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = self._nqueries[working_ind]
            for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
                self.distortion_all[index_over_all_images][self._nqueries[inside_batch_index].item()] = dist[
                    inside_batch_index].item()

            log.info('Attacking image {} - {} / {}, distortion {}, query {}'.format(
                batch_index * args.batch_size, (batch_index + 1) * args.batch_size, self.total_images,
                dist.cpu().numpy(), self._nqueries.cpu().numpy()))

        success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return best_advs, self._nqueries, success_stop_queries, dist, (dist <= self.epsilon)


    def attack_all_images(self, args, arch_name, result_dump_path, config):
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
            # if correct.int().item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
            #     log.info("{}-th original image is classified incorrectly, skip!".format(batch_index + 1))
            #     continue
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
                    assert target_labels[0].item() != true_labels[0].item()
                    # log.info("load random label as {}".format(target_labels))
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                # target_images = self.get_image_of_target_class(self.dataset_name, target_labels, self.model)
                target_images = select_random_image_of_target_class(self.dataset_name, target_labels, self.model, self.load_random_class_image)
                if target_images is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
            else:
                target_labels = None
                target_images = None

            adv_images, query, success_query, distortion_with_max_queries, success_epsilon = self.attack(batch_index, images, target_images, true_labels, target_labels, **config["run"])

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
                          "success_all": self.success_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "success_query_all": self.success_query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion_all,
                          "avg_distortion_with_max_queries": self.distortion_with_max_queries_all.mean().item(),
                          "args": vars(args)}
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_dump_path))


class Basis:
    def __init__(self, originals, random_noise: str = "normal", basis_type: str = "dct", **kwargs):
        """
        Args:
            random_noise (str, optional): When basis is created, a noise will be added.This noise can be normal or
                                          uniform. Defaults to "normal".
            basis_type (str, optional): Type of the basis: DCT, Random, Genetic,. Defaults to "random".
            device (int, optional): [description]. Defaults to -1.
            args, kwargs: In args and kwargs, there is the basis params:
                    * Random: No parameters
                    * DCT:
                            * function (tanh / constant / linear): function applied on the dct
                            * alpha
                            * beta
                            * lambda
                            * frequence_range: integers or float
                            * min_dct_value
                            * dct_type: 8x8 or full
        """
        self._originals = originals
        self._direction_shape = originals.shape[1:]
        self.basis_type = basis_type

        self._load_params(**kwargs)

        assert random_noise in ["normal", "uniform"]
        self.random_noise = random_noise

    def get_vector(self, ortho_with = None, bounds = (0, 1)):
        if ortho_with is None:
            ortho_with = {i: None for i in range(len(self._originals))}

        vectors = [
            self.get_vector_i(i, ortho_with[i], bounds)
            for i in range(len(self._originals))
        ]
        return torch.cat(vectors, dim=0)

    def get_vector_i(self, index, ortho_with = None, bounds = (0, 1)):
        r: torch.Tensor = getattr(self, "_get_vector_i_" + self.basis_type)(index, bounds).to(self._originals.device)
        if ortho_with is not None:
            r_repeated = torch.cat([r.unsqueeze(0)] * len(ortho_with), dim=0).to(self._originals.device)

            # inner product
            gs_coeff = (ortho_with * r_repeated).flatten(1).sum(1)
            proj = atleast_kdim(gs_coeff, len(ortho_with.shape)) * ortho_with
            r = r - proj.sum(0)
        r = r.unsqueeze(0)
        return r / atleast_kdim(r.flatten(1).norm(dim=1), len(r.shape))

    def _get_vector_i_dct(self, index, bounds):
        r_np = np.zeros(self._direction_shape)
        for channel, dct_channel in enumerate(self.dcts[index]):
            probs = np.random.randint(-2, 1, dct_channel.shape) + 1
            r_np[channel] = dct_channel * probs
        r_np = idct2_full(r_np) + self._beta * (2 * np.random.rand(*r_np.shape) - 1)
        return torch.from_numpy(r_np.astype("float32"))

    def _get_vector_i_random(self, index, bounds):
        r = torch.zeros_like(self._originals)
        r = getattr(torch, self.random_noise)(r, r.shape, *bounds)
        return r

    def _load_params(
            self,
            beta = 0,
            frequence_range = (0, 1),
            dct_type = "full",
            function = "tanh",
            lambda_ = 1
    ) -> None:
        if not hasattr(self, "_get_vector_i_" + self.basis_type):
            raise ValueError("Basis {} doesn't exist.".format(self.basis_type))

        if self.basis_type == "dct":
            self._beta = beta
            if dct_type == "8x8":
                mask_size = (8, 8)
                dct_function = dct2_8_8
            elif dct_type == "full":
                mask_size = (self._direction_shape[-2], self._direction_shape[-1])
                dct_function = dct2_full
            else:
                raise ValueError("DCT {} doesn't exist.".format(dct_type))

            dct_mask = get_zig_zag_mask(frequence_range, mask_size)
            self.dcts = np.array([dct_function(np.array(image.cpu()), dct_mask) for image in self._originals])

            def get_function(function, lambda_):
                if function == "tanh":
                    return lambda x: np.tanh(lambda_ * x)
                elif function == "identity":
                    return lambda x: x
                elif function == "constant":
                    return lambda x: (abs(x) > 0).astype(int)
                else:
                    raise ValueError("Function given for DCT is incorrect.")

            self.dcts = get_function(function, lambda_)(self.dcts)


def dct2(a):
    return fft.dct(fft.dct(a, axis=0), axis=1)


def idct2(a):
    return fft.idct(fft.idct(a, axis=0), axis=1)


def dct2_8_8(image, mask = None):
    if mask is None:
        mask = np.ones((8, 8))
    if mask.shape != (8, 8):
        raise ValueError("Mask have to be with a size of (8, 8)")

    imsize = image.shape
    dct = np.zeros_like(image)

    for channel in range(imsize[0]):
        for i in np.r_[:imsize[1]:8]:
            for j in np.r_[:imsize[2]:8]:
                dct_i_j = dct2(image[channel, i:(i + 8), j:(j + 8)])
                dct[channel, i:(i + 8), j:(j + 8)] = dct_i_j * mask[:dct_i_j.shape[0], :dct_i_j.shape[1]]
    return dct


def idct2_8_8(dct):
    im_dct = np.zeros(dct.shape)

    for channel in range(dct.shape[0]):
        for i in np.r_[:dct.shape[1]:8]:
            for j in np.r_[:dct.shape[2]:8]:
                im_dct[channel, i:(i + 8), j:(j + 8)] = idct2(dct[channel, i:(i + 8), j:(j + 8)])
    return im_dct


def dct2_full(image, mask = None):
    if mask is None:
        mask = np.ones(image.shape[-2:])

    imsize = image.shape
    dct = np.zeros(imsize)

    for channel in range(imsize[0]):
        dct_i_j = dct2(image[channel])
        dct[channel] = dct_i_j * mask
    return dct


def idct2_full(dct):
    im_dct = np.zeros(dct.shape)

    for channel in range(dct.shape[0]):
        im_dct[channel] = idct2(dct[channel])
    return im_dct


def get_zig_zag_mask(frequence_range, mask_shape = (8, 8)):
    mask = np.zeros(mask_shape)
    s = 0
    total_component = sum(mask.flatten().shape)

    if frequence_range[1] <= 1:
        n_coeff = int(total_component * frequence_range[1])
    else:
        n_coeff = int(frequence_range[1])

    if frequence_range[0] <= 1:
        min_coeff = int(total_component * frequence_range[0])
    else:
        min_coeff = int(frequence_range[0])

    while n_coeff > 0:
        for i in range(min(s + 1, mask_shape[0])):
            for j in range(min(s + 1, mask_shape[1])):
                if i + j == s:
                    if min_coeff > 0:
                        min_coeff -= 1
                        continue

                    if s % 2:
                        mask[i, j] = 1
                    else:
                        mask[j, i] = 1
                    n_coeff -= 1
                    if n_coeff == 0:
                        return mask
        s += 1
    return mask


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.attack_defense:
        dirname = 'SurFree_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
    else:
        dirname = 'SurFree-{}-{}-{}'.format(dataset, norm, target_str)
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
    parser.add_argument('--json-config', type=str, default='./configures/SurFree.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    parser.add_argument("--norm",type=str, choices=["l2","linf"],required=True)
    parser.add_argument('--batch-size', type=int, default=1)
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

    args = parser.parse_args()
    # assert args.batch_size == 1, "Triangle Attack only supports mini-batch size equals 1!"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
        config = {"init": {}, "run": {"epsilons": None}}
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        config = json.load(open(args.json_config, "r"))
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
        attacker = SurFree(model, args.dataset, 0, 1.0, model.input_size[-2], model.input_size[-1],
                           IN_CHANNELS[args.dataset], args.batch_size, args.epsilon, args.norm, args.load_random_class_image,
                           maximum_queries=args.max_queries, **config["init"])
        attacker.attack_all_images(args, arch, save_result_path, config)
        model.cpu()
