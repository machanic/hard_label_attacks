from collections import OrderedDict, defaultdict

import json
import torch
from torch.nn import functional as F
import numpy as np
import glog as log
from config import CLASS_NUM, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, TinyImageNetDataset
from utils.dataset_toolkit import select_random_image_of_target_class

class PriorOptLinfNorm(object):
    def __init__(self, model, surrogate_models, dataset, epsilon, targeted, batch_size=1, k=100, alpha=0.2, beta=0.001, iterations=1000,
                 maximum_queries=10000, sign=False, momentum=0.0, clip_grad_max_norm=1.0, tol=None, prior_grad_binary_search_tol=0.01,
                 best_initial_target_sample=False):
        self.model = model
        self.surrogate_models = surrogate_models
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.maximum_queries = maximum_queries
        self.sign = sign
        self.momentum = momentum
        self.epsilon = epsilon
        self.targeted = targeted
        self.best_initial_target_sample = best_initial_target_sample
        self.dataset = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, model.arch)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)

        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)
        self.clip_grad_max_norm = clip_grad_max_norm
        self.tol = tol
        self.prior_grad_binary_search_tol = prior_grad_binary_search_tol

    def convert_linf_to_l2_distance(self, theta, distance_linf):
        return distance_linf / torch.norm(theta.view(-1), p=float('inf')) * torch.norm(theta.view(-1), p=2)

    def convert_l2_to_linf_distance(self, theta, distance_l2):
        return torch.norm(distance_l2 / torch.norm(theta.view(-1), p=2) * theta.view(-1), p=float('inf'))

    def norm_l2(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def clip_grad_norm(self, grad:torch.tensor, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        r"""Clips gradient norm.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        device = grad.device
        if norm_type == float('inf'):
            total_norm = grad.detach().view(-1).abs().max().to(device)
        else:
            total_norm = torch.norm(grad.detach().view(-1), norm_type).to(device)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grad.mul_(clip_coef.to(device))
        return grad

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, max_high_bound=20, tol=1e-5):
        nquery = 1
        lbd = self.convert_linf_to_l2_distance(theta, initial_lbd)  # 传入就是个无穷范数距离,转为2范数

        # still inside boundary
        if model(x0 + lbd * theta).max(1)[1].item() == y0:
            if lbd > max_high_bound:
                max_high_bound = lbd + 50
                log.warn("warn: lbd > max_high_bound, reset max_high_bound to {}".format(max_high_bound))
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while model(x0 + lbd_hi * theta).max(1)[1].item() == y0:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > max_high_bound:
                    return float('inf'), nquery - 1
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while model(x0 + lbd_lo * theta).max(1)[1].item() != y0:
                lbd_lo = lbd_lo * 0.99
                nquery += 1
        tot_count = 0
        old_lbd_mid = lbd_hi
        while (lbd_hi - lbd_lo) > tol:
            tot_count+=1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model(x0 + lbd_mid * theta).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if old_lbd_mid == lbd_mid or tot_count > 200:
                log.warn(
                    "binary search's lowest numerical precision warn: tol is {:.2e} and the while loop is executed {} times, break!".format(
                        tol, tot_count))
                break
            old_lbd_mid = lbd_mid
        lbd_hi = self.convert_l2_to_linf_distance(theta, lbd_hi) # 转为无穷范数距离
        return lbd_hi, nquery

    def fine_grained_binary_search_local_targeted(self, model, x0, t, theta, initial_lbd=1.0, max_high_bound=100, tol=1e-5):
        nquery = 1
        lbd = self.convert_linf_to_l2_distance(theta, initial_lbd)

        if model(x0 + lbd * theta).max(1)[1].item() != t:
            if lbd > max_high_bound:
                max_high_bound = lbd + 50
                log.warn("warn: lbd > max_high_bound, reset max_high_bound to {}".format(max_high_bound))
            lbd_lo = lbd
            lbd_hi = lbd * 1.01
            nquery += 1
            while model(x0 + lbd_hi * theta).max(1)[1].item() != t:
                lbd_hi = lbd_hi * 1.01
                nquery += 1
                if lbd_hi > max_high_bound:
                    return float('inf'), nquery - 1
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.99
            nquery += 1
            while model(x0 + lbd_lo * theta).max(1)[1].item() == t:
                lbd_lo = lbd_lo * 0.99
                nquery += 1
        tot_count = 0
        old_lbd_mid = lbd_hi
        while (lbd_hi - lbd_lo) > tol:
            tot_count += 1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            if model(x0 + lbd_mid * theta).max(1)[1].item() == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if old_lbd_mid == lbd_mid or tot_count > 200:
                log.warn(
                    "binary search's lowest numerical precision warn: tol is {:.2e} and the while loop is executed {} times, break!".format(
                        tol, tot_count))
                break
            old_lbd_mid = lbd_mid
        lbd_hi = self.convert_l2_to_linf_distance(theta, lbd_hi) # 转为无穷范数距离
        return lbd_hi, nquery

    def fine_grained_binary_search(self,  x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        initial_lbd = self.convert_linf_to_l2_distance(theta, initial_lbd)
        current_best = self.convert_linf_to_l2_distance(theta, current_best)
        if initial_lbd > current_best:
            nquery += 1
            if self.model(x0 + current_best * theta).max(1)[1].item() == y0:
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            count += 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if count >= 200:
                log.info("Break in the first fine_grained_binary_search!")
                break
        lbd_hi = self.convert_l2_to_linf_distance(theta, lbd_hi) # 转为无穷范数距离
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
        initial_lbd = self.convert_linf_to_l2_distance(theta, initial_lbd)
        current_best = self.convert_linf_to_l2_distance(theta, current_best)
        if initial_lbd > current_best:
            nquery += 1
            if self.model(x0 + current_best * theta).max(1)[1].item() != t:
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd

        lbd_hi = lbd
        lbd_lo = 0.0
        count = 0
        while (lbd_hi - lbd_lo) > 1e-3:  # was 1e-5
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            nquery += 1
            count += 1
            if self.model(x0 + lbd_mid * theta).max(1)[1].item() != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
            if count >= 200:
                log.info("Break in the first fine_grained_binary_search!")
                break
        lbd_hi = self.convert_l2_to_linf_distance(theta, lbd_hi)  # 转为无穷范数距离
        return lbd_hi, nquery

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            _, argsort = logit.sort(dim=1, descending=True)
            # print('True label:{}, The max label:{}, Second max label:{}'.format(label.item(), argsort[:, 0].item(), argsort[:, 1].item()))
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def g_function_bin_search(self, model, image, theta, initial_lbd, true_labels, target_labels, tol=1e-3):

        initial_lbd = self.convert_linf_to_l2_distance(theta, initial_lbd)
        if target_labels is not None:
            start = initial_lbd.item() + 1.0
            # assert start <= 100.0, "initial_lbd > 100 error! It is {}".format(start)
            target_label = model((image + start * theta).cuda()).max(1)[1].item()
            num_points = 400
            lmdb_list = torch.linspace(start, 0, num_points).view(-1, 1, 1, 1).to(image.device)
            perturbed = image + lmdb_list * theta
            pred_labels = model(perturbed.cuda()).max(1)[1]
            any_non_targets = pred_labels != target_label
            if any_non_targets.any().item():
                target_labels = target_label
                initial_lbd = lmdb_list[any_non_targets][0].item()  # binary search will gradually increase the bound distance to find target class
            else:
                true_label = model(image.cuda()).max(1)[1].item()
                lmdb_list = torch.linspace(0, 200, num_points).view(-1, 1, 1, 1).to(image.device)
                perturbed = image + lmdb_list * theta
                pred_labels = model(perturbed.cuda()).max(1)[1]
                any_non_true = pred_labels != true_label
                if any_non_true.any().item():
                    target_labels = pred_labels[any_non_true][0].item()
                    # 有一种可能性是pred_labels==true_label得出的bool的tensor为[True,True,...,True,False,...,False,True,True,...]
                    # 为了得到第一段的最后一个True的位置，则要使用下面的方法
                    true_label_array = ~any_non_true.view(-1)
                    diff = true_label_array[:-1] ^ true_label_array[1:]
                    diff = diff.int()
                    # diff = torch.diff((~any_non_true.view(-1)).int()) TODO 新版PyTorch改为torch.diff
                    indices = torch.nonzero(diff).squeeze(0)
                    index_of_last_true = indices[0].item()
                    initial_lbd = lmdb_list[index_of_last_true].item()

        max_high_bound = initial_lbd + 100
        initial_lbd = self.convert_l2_to_linf_distance(theta, initial_lbd)
        if target_labels is None:
            lmdb_result, nquery = self.fine_grained_binary_search_local(model, image, true_labels, theta, initial_lbd,
                                                                        max_high_bound=max_high_bound, tol=tol)
        else:
            lmdb_result, nquery = self.fine_grained_binary_search_local_targeted(model, image, target_labels, theta, initial_lbd,
                                                                        max_high_bound=max_high_bound, tol=tol)

        if lmdb_result == float("inf"):
            log.warn("warn: float('inf') value of the distance along theta in get_grad of surrogate model!")
            lmdb_result = initial_lbd
        return lmdb_result, target_labels,  nquery

    def get_g_grad(self, model, images, theta, initial_lbd, true_labels, target_labels):
        if images.size(-1) != model.input_size[-1]:
            images = F.interpolate(images, size=model.input_size[-1], mode='bilinear', align_corners=False)
            theta = F.interpolate(theta, size=model.input_size[-1], mode='bilinear', align_corners=False)
            theta /= self.norm_l2(theta)
        with torch.no_grad():
            min_lmdb, target_labels, _ = self.g_function_bin_search(model, images, theta.detach(), initial_lbd, true_labels, target_labels)
        if target_labels is not None:
            target_labels = torch.tensor([target_labels]).long().cuda()

        with torch.enable_grad():
            theta.requires_grad_()
            loss = self.cw_loss(model(images + self.convert_linf_to_l2_distance(theta, min_lmdb).item()
                                 * theta / torch.norm(theta, p=2, dim=(1, 2, 3), keepdim=True)),
                                true_labels, target_labels)
            grad_theta = torch.autograd.grad(-loss, theta, create_graph=False)[0]
        return grad_theta

    def prior_grad(self, images, theta, initial_lbd, true_label, target_label=None, sigma=0.001):
        assert images.dim() == 4
        assert theta.dim() == 4
        query = 0
        prior_grads = []
        initial_lbd_l2_norm = self.convert_linf_to_l2_distance(theta, initial_lbd)
        for surrogate_model in self.surrogate_models:
            prior_grad = self.get_g_grad(surrogate_model, images, theta, initial_lbd, true_label, target_label)
            assert not torch.isnan(prior_grad.sum()), "prior grad is nan"
            prior_grad = prior_grad / torch.norm(prior_grad, p=2, dim=(1, 2, 3), keepdim=True)
            prior_grads.append(prior_grad)

        us = []
        for prior_grad in prior_grads:
            us.append(prior_grad.squeeze())
        for i in range(self.k - len(prior_grads)):
            rv = torch.randn_like(theta.squeeze())
            rv = rv / torch.norm(rv.view(-1), p=2, dim=0)
            us.append(rv)
        # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
        orthos = []
        for u in us:
            for ou in orthos:
                u = u - torch.sum(u * ou) * ou
            u = u / torch.sqrt(torch.sum(u * u))
            orthos.append(u)
        # perform sign-based RGF gradient estimation
        images_batch = []
        u_batch = []
        for rv in orthos[len(prior_grads):]:
            u = rv.unsqueeze(0)
            new_theta = theta + sigma * u
            new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
            u_batch.append(u)
            images_batch.append(images + initial_lbd_l2_norm * new_theta)  # No matter which the new_theta is, we need the same L2 norm radius!
            query += 1
        images_batch = torch.cat(images_batch, 0)
        u_batch = torch.cat(u_batch, 0)  # B,C,H,W
        assert u_batch.dim() == 4
        assert u_batch.size(0) == self.k - len(prior_grads)
        sign = torch.ones(self.k - len(prior_grads), device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(self.k - len(prior_grads))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(self.k - len(prior_grads))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels != true_labels] = -1
        sign_grad = torch.sum(u_batch * sign.view(self.k - len(prior_grads), 1, 1, 1), dim=0, keepdim=True)
        sign_grad /= (self.k - len(prior_grads))
        # perform score-based RGF gradient estimation, and we need to perform Gram-Schmidt orthogonalization again.
        sign_grad_ortho = sign_grad - sum(torch.sum(sign_grad * ou) * ou for ou in orthos[0:len(prior_grads)])
        sign_grad_ortho = sign_grad_ortho / torch.sqrt(torch.sum(sign_grad_ortho * sign_grad_ortho))
        new_orthos = orthos[0:len(prior_grads)] + [sign_grad_ortho]
        est_grad = torch.zeros_like(theta)
        derivatives = []
        for grad_theta_orth in new_orthos:
            new_theta = theta + sigma * grad_theta_orth
            new_theta /= self.norm_l2(new_theta)
            initial_lbd_l2_norm = self.convert_linf_to_l2_distance(new_theta, initial_lbd)
            max_high_bound = 100
            if initial_lbd_l2_norm > max_high_bound:
                max_high_bound = initial_lbd_l2_norm + 100
            if true_label is not None:
                perturb_bound, bs_count = self.fine_grained_binary_search_local(self.model, images, true_label, new_theta,
                                                                              initial_lbd, max_high_bound, tol=self.prior_grad_binary_search_tol)
            else:
                perturb_bound, bs_count = self.fine_grained_binary_search_local_targeted(self.model, images, target_label,
                                                                                       new_theta, initial_lbd, max_high_bound,
                                                                                       tol=self.prior_grad_binary_search_tol)
            query += bs_count
            if perturb_bound == float("inf"):
                log.warn("warn: the returned boundary distance is float('inf') after the binary search for calculating the loss derivative.")
                continue
                # assert prior_bound!=float('inf'), "Error! returned float('inf') in binary search for calculating the loss derivative."
            weight = (perturb_bound - initial_lbd) / sigma  # 梯度指的是朝着lmdb边界距离上升的方向
            derivatives.append(weight)
            est_grad += weight * grad_theta_orth
        if len(derivatives) > 0:
            est_grad = est_grad / len(derivatives)
        return est_grad, query


    def prior_sign_grad(self, images, theta, initial_lbd, true_label, target_label=None, sigma=0.001):
        assert images.dim()==4
        assert theta.dim()==4
        query = 0
        prior_grads = []
        initial_lbd_l2_norm = self.convert_linf_to_l2_distance(theta, initial_lbd)
        for surrogate_model in self.surrogate_models:
            prior_grad = self.get_g_grad(surrogate_model, images, theta, initial_lbd, true_label, target_label)
            prior_grad = prior_grad / torch.norm(prior_grad, p=2, dim=(1, 2, 3), keepdim=True)
            prior_grads.append(prior_grad)

        us = []
        for prior_grad in prior_grads:
            us.append(prior_grad.squeeze())
        for i in range(self.k - len(prior_grads)):
            rv = torch.randn_like(theta.squeeze())
            rv = rv / torch.norm(rv.view(-1), p=2, dim=0)
            us.append(rv)
        # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
        orthos = []
        for u in us:
            for ou in orthos:
                u = u - torch.sum(u * ou) * ou
            u = u / torch.sqrt(torch.sum(u * u))
            orthos.append(u)
        orthos = torch.stack(orthos, dim=0)

        images_batch = []
        u_batch = []
        for orth in orthos:
            u = orth.unsqueeze(0)
            new_theta = theta + sigma * u
            new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
            u_batch.append(u)
            images_batch.append(images + initial_lbd_l2_norm * new_theta)  # No matter which the new_theta is, we need the same L2 norm radius!
            query += 1
        images_batch = torch.cat(images_batch, 0)
        u_batch = torch.cat(u_batch, 0)  # B,C,H,W
        assert u_batch.dim() == 4
        sign = torch.ones(orthos.size(0), device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(orthos.size(0))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(orthos.size(0))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels != true_labels] = -1
        sign_grad = torch.sum(u_batch * sign.view(orthos.size(0), 1, 1, 1), dim=0, keepdim=True)

        sign_grad = sign_grad / orthos.size(0)

        return sign_grad, query


    def untargeted_attack(self, image_index, images, true_labels):
        assert images.size(0) == 1
        alpha = self.alpha
        beta = self.beta
        momentum = self.momentum
        batch_image_positions = np.arange(image_index * self.batch_size,
                                          min((image_index + 1) * self.batch_size, self.total_images)).tolist()
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        ls_total = 0
        true_label = true_labels[0].item()
        # Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        log.info("Searching for the initial direction on {} random directions.".format(num_directions))
        for i in range(num_directions):
            query += 1
            theta = torch.randn_like(images)
            if self.model(images + self.convert_linf_to_l2_distance(theta, 1.0) * theta).max(1)[1].item() != true_label:
                initial_lbd = torch.norm(theta.view(-1),p=float('inf'))
                theta /= self.norm_l2(theta)
                lbd, count = self.fine_grained_binary_search(images, true_label, theta, initial_lbd, g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    log.info("{}-th image, {}-th iteration distortion: {:.4f}".format(image_index + 1, i, g_theta))
                    self.count_stop_query_and_distortion(images, images + best_theta * self.convert_linf_to_l2_distance(best_theta, g_theta), query, success_stop_queries,
                                                         batch_image_positions)
        ## fail if cannot find an adversarial direction within 200 Gaussian
        if g_theta == float('inf'):
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index + 1))
            return images, query, success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index + 1, g_theta, query[0].item()))
        #### Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        vg = torch.zeros_like(xg)
        for i in range(self.iterations):
            ## gradient estimation at x0 + theta (init)
            if self.sign:
                grad, grad_queries = self.prior_sign_grad(images, xg, gg, true_label, None, beta)
            else:
                grad, grad_queries = self.prior_grad(images, xg, gg, true_label, None, beta)
            grad = self.clip_grad_norm(grad, max_norm=self.clip_grad_max_norm)
            ## Line search of the step size of gradient descent
            query += grad_queries
            ls_count = 0  # line search queries
            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta
            min_vg = vg  ## velocity (for momentum only)
            for _ in range(15):
                # update theta by one step sgd
                if momentum > 0:
                    new_vg = momentum * vg - alpha * grad
                    new_theta = xg + new_vg
                else:
                    new_theta = xg - alpha * grad
                new_theta /= torch.norm(new_theta.view(-1),p=2)
                tol = beta/500
                if self.tol is not None:
                    tol = self.tol
                new_g2, count = self.fine_grained_binary_search_local(self.model, images, true_label, new_theta,
                                                                      initial_lbd=min_g2, tol=tol)
                ls_count += count
                query += count
                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    self.count_stop_query_and_distortion(images, images + min_theta * self.convert_linf_to_l2_distance(min_theta, min_g2), query,
                                                         success_stop_queries, batch_image_positions)
                    if momentum > 0:
                        min_vg = new_vg
                else:
                    break
            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    if momentum > 0:
                        new_vg = momentum * vg - alpha * grad
                        new_theta = xg + new_vg
                    else:
                        new_theta = xg - alpha * grad
                    new_theta /= torch.norm(new_theta)
                    tol = beta / 500
                    if self.tol is not None:
                        tol = self.tol
                    new_g2, count = self.fine_grained_binary_search_local(self.model, images, true_label, new_theta,
                                                                          initial_lbd=min_g2, tol=tol)
                    ls_count += count
                    query += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        self.count_stop_query_and_distortion(images, images + min_theta * self.convert_linf_to_l2_distance(min_theta, min_g2), query,
                                                             success_stop_queries, batch_image_positions)
                        if momentum > 0:
                            min_vg = new_vg
                        break
            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                log.info("{}-th image warns: not moving".format(image_index+1))
                beta = beta * 0.1
                # if beta < 1e-8 and self.tol is None:
                #     break
                beta = max(beta, 1e-8)
            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2
            vg = min_vg

            ls_total += ls_count
            ## logging
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index+1, i+1, gg, query[0].item()))
            if query.min().item() >= self.maximum_queries:
                break
        if self.epsilon is None or gg <= self.epsilon:
            target = self.model(images + self.convert_linf_to_l2_distance(xg, gg) * xg).max(1)[1].item()
            log.info("{}-th image success distortion {:.4f} target {} queries {} LS queries {}".format(image_index+1,
                                                                                                       gg, target, query[0].item(), ls_total))
        # gg 是distortion
        distortion = torch.norm(self.convert_linf_to_l2_distance(xg,gg) * xg, p=float('inf'))
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        return images + self.convert_linf_to_l2_distance(xg, gg) * xg, query,success_stop_queries, torch.tensor([gg]).float(), torch.tensor([gg]).float() <= self.epsilon, xg

    def targeted_attack(self, image_index, images, target_labels, target_class_image):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        target_label = target_labels[0].item()

        if (self.model(images).max(1)[1].item() == target_label):
            log.info("{}=th image is already predicted as target label! No need to attack.".format(image_index+1))

        alpha = self.alpha
        beta = self.beta
        batch_image_positions = np.arange(image_index * self.batch_size,
                                          min((image_index + 1) * self.batch_size, self.total_images)).tolist()
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        ls_total = 0

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        log.info("Searching for the initial direction on {} samples: ".format(num_samples))
        if self.best_initial_target_sample:
            # Iterate through training dataset. Find best initial point for gradient descent.
            if self.dataset == "ImageNet":
                val_dataset = ImageNetDataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            elif self.dataset == "CIFAR-10":
                val_dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            elif self.dataset == "CIFAR-100":
                val_dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[self.dataset], target_label, "validation")
            val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)
            for i, (xi, yi) in enumerate(val_dataset_loader):
                if self.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                    xi = F.interpolate(xi,
                                           size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                           align_corners=False)
                xi = xi.cuda()
                yi_pred = self.model(xi).max(1)[1].item()
                query += 1
                if yi_pred != target_label:
                    continue

                theta = xi - images
                theta /= self.norm_l2(theta)
                initial_lbd = torch.norm(theta.view(-1), p=float('inf'))
                lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd,
                                                                      g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.count_stop_query_and_distortion(images, images + best_theta * self.convert_linf_to_l2_distance(best_theta, g_theta), query,
                                                         success_stop_queries, batch_image_positions)
                    log.info("{}-th image. Found initial target image with the distortion {:.4f}".format(image_index+1, g_theta))

                if i > 100:
                    break
        else:
            # xi = self.get_image_of_target_class(self.dataset, target_labels, self.model)
            xi = target_class_image
            theta = xi - images
            initial_lbd = torch.norm(theta.view(-1), p=float('inf'))
            theta /= self.norm_l2(theta)
            lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd,
                                                                  g_theta)
            query += count
            best_theta, g_theta = theta, lbd
            self.count_stop_query_and_distortion(images, images + best_theta * self.convert_linf_to_l2_distance(best_theta, g_theta), query,
                                                 success_stop_queries, batch_image_positions)
        if g_theta == np.inf:
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index + 1))
            return images, query, success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index + 1, g_theta,
                                                                                    query[0].item()))
        # Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        for i in range(self.iterations):
            if self.sign:
                grad, grad_queries = self.prior_sign_grad(images, xg, gg, None, target_label, beta)
            else:
                grad, grad_queries = self.prior_grad(images, xg, gg, None, target_label, beta)
            grad = self.clip_grad_norm(grad, max_norm=self.clip_grad_max_norm)
            query += grad_queries
            # Line search
            ls_count = 0
            min_theta = xg
            min_g2 = gg
            for _ in range(15):
                new_theta = xg - alpha * grad
                new_theta /= torch.norm(new_theta.view(-1), p=2)
                tol = beta / 500
                if self.tol is not None:
                    tol = self.tol
                new_g2, count = self.fine_grained_binary_search_local_targeted(self.model, images, target_label, new_theta, initial_lbd=min_g2, tol=tol)
                ls_count += count
                query += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    self.count_stop_query_and_distortion(images, images + min_theta * self.convert_linf_to_l2_distance(min_theta, min_g2), query,
                                                         success_stop_queries, batch_image_positions)
                else:
                    break

            if min_g2 >= gg:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * grad
                    new_theta /= torch.norm(new_theta.view(-1), p=2)
                    tol = beta / 500
                    if self.tol is not None:
                        tol = self.tol
                    new_g2, count = self.fine_grained_binary_search_local_targeted(self.model, images, target_label,
                                                                                   new_theta, initial_lbd=min_g2, tol=tol)
                    ls_count += count
                    query += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        self.count_stop_query_and_distortion(images, images + min_theta * self.convert_linf_to_l2_distance(min_theta, min_g2), query,
                                                             success_stop_queries, batch_image_positions)
                        break

            if alpha < 1e-4:
                alpha = 1.0
                log.info("{}-th image, warning: not moving".format(image_index+1))
                beta = beta * 0.1
                # if beta < 1e-8 and self.tol is None:
                #     break
                beta = max(beta, 1e-8)
            xg, gg = min_theta, min_g2

            ls_total += ls_count
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index + 1, i + 1, gg,
                                                                                           query[0].item()))
            if query.min().item() >= self.maximum_queries:
                break

        log.info(
            "{}-th image success distortion {:.4f} queries {} stop queries {}".format(image_index + 1,
                                                                                              gg,
                                                                                              query[0].item(),
                                                                                              success_stop_queries[0].item()))

        adv_target = self.model(images + self.convert_linf_to_l2_distance(xg, gg) * xg).max(1)[1].item()
        if adv_target == target_label:
            log.info("{}-th image attack successfully! Distortion {:.4f} target {} queries:{} success stop queries:{} LS queries:{}".format(image_index + 1,
                                                                                                       gg, adv_target,
                                                                                                       query[0].item(), success_stop_queries[0].item(),
                                                                                                       ls_total))
        else:
            log.info("{}-th image is failed to find targeted adversarial example.".format(image_index+1))

        distortion = torch.norm(self.convert_linf_to_l2_distance(xg,gg) * xg, p=float('inf'))
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        return images + self.convert_linf_to_l2_distance(xg, gg) * xg, query, success_stop_queries, torch.tensor([gg]).float(), torch.tensor(
            [gg]).float() <= self.epsilon, xg



    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):
        dist = torch.norm((perturbed - images).view(images.size(0), -1), p=float('inf'), dim=1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1)
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

    def attack_all_images(self, args, arch_name, result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images,
                                       size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear',
                                       align_corners=False)
            images = images.cuda()
            with torch.no_grad():
                logit = self.model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels.cuda()).float()  # shape = (batch_size,)
            if correct.int().item() == 0: # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                log.info("{}-th original image is classified incorrectly, skip!".format(batch_index+1))
                continue

            all_surrogate_correct = True
            for surrogate_model in self.surrogate_models:
                with torch.no_grad():
                    logit_surrogate = surrogate_model(images)
                pred_surrogate = logit_surrogate.argmax(dim=1)
                correct_surrogate = pred_surrogate.eq(true_labels.cuda()).float()  # shape = (batch_size,)
                if correct_surrogate.int().item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                    log.info("{}-th original image is classified incorrectly by surrogate model, skip!".format(batch_index + 1))
                    all_surrogate_correct = False
                    break
            if not all_surrogate_correct:
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
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            else:
                target_labels = None
            # return images + gg * xg, query,success_stop_queries, gg, gg <= self.epsilon, xg
            # adv, distortion, is_success, nqueries, theta_signopt
            if args.targeted:
                # target_class_image = self.get_image_of_target_class(self.dataset, target_labels, self.model)
                target_class_image = select_random_image_of_target_class(self.dataset, target_labels, self.model, args.load_random_class_image)
                if target_class_image is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
                target_class_image = target_class_image.cuda()
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta_signopt = self.targeted_attack(batch_index, images, target_labels, target_class_image)
            else:
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta_signopt = self.untargeted_attack(batch_index,
                                                                                                                    images,  true_labels)
            distortion_with_max_queries = distortion_with_max_queries.detach().cpu()

            with torch.no_grad():
                adv_logit = self.model(adv_images.cuda())
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * success_epsilon.float() *(success_query <= self.maximum_queries).float()

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
