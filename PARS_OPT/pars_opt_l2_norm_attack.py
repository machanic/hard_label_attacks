from collections import OrderedDict, defaultdict

import random
import json
import torch
from torch.nn import functional as F
import numpy as np
import glog as log
from config import CLASS_NUM, IMAGE_DATA_ROOT
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset
from utils.dataset_toolkit import select_random_image_of_target_class


class ParsOptL2Norm(object):
    def __init__(self, model, surrogate_models, dataset, epsilon, targeted, batch_size=1, k=100, alpha=0.2,
                 beta=0.001, iterations=1000, maximum_queries=10000, method='ARS', clip_grad_max_norm=1.0, tol=None,
                 grad_binary_search_tol=None, best_initial_target_sample=False, resized=False, dim_reduce_factor=4,
                 line_search=False):
        self.model = model
        self.surrogate_models = surrogate_models if method == 'PARS' else []
        self.dataset = dataset
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, model.arch)
        self.batch_size = batch_size
        self.total_images = len(self.dataset_loader.dataset)
        self.epsilon = epsilon
        self.iterations = iterations
        self.maximum_queries = maximum_queries
        self.targeted = targeted
        self.best_initial_target_sample = best_initial_target_sample

        self.method = method
        self.q = k
        self.alpha = alpha
        self.beta = beta
        self.clip_grad_max_norm = clip_grad_max_norm
        self.tol = tol
        self.prior_grad_binary_search_tol = grad_binary_search_tol
        self.resized = resized
        self.line_search = line_search
        self.dim_reduce_factor = dim_reduce_factor

        self.query_all = torch.zeros(self.total_images)
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)


    def norm(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def clip_grad_norm(self, grad:torch.tensor, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        r"""Clips gradient norm of an image gradient.

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

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd=1.0, max_high_bound=100, tol=1e-5, search_bound=True):
        # search_bound = False
        nquery = 1
        lbd = initial_lbd.item() if isinstance(initial_lbd, torch.Tensor) else initial_lbd

        # still inside boundary
        if model(x0 + lbd * theta).max(1)[1].item() == y0:
            if not search_bound:
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
                        return float('inf'), nquery
            else:
                return float('inf'), nquery
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
            tot_count += 1
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
        return lbd_hi, nquery

    def fine_grained_binary_search_local_targeted(self, model, x0, t, theta, initial_lbd=1.0, max_high_bound=100, tol=1e-5, search_bound=True):
        # search_bound = False
        nquery = 1
        lbd = initial_lbd.item() if isinstance(initial_lbd, torch.Tensor) else initial_lbd

        if model(x0 + lbd * theta).max(1)[1].item() != t:
            if not search_bound:
                if lbd > max_high_bound:
                    max_high_bound = lbd + 50
                    log.warn("warn: lbd > max_high_bound, reset max_high_bound to {}".format(max_high_bound))
                lbd_lo = lbd
                lbd_hi = lbd * 1.01
                nquery += 1
                while model(x0 + lbd_hi * theta).max(1)[1].item() != t:
                    lbd_hi = lbd_hi * 1.01
                    nquery += 1
                    if lbd_hi > max_high_bound or nquery > 200:
                        return float('inf'), nquery
            else:
                return float('inf'), nquery
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
        return lbd_hi, nquery

    def fine_grained_binary_search(self, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
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
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, x0, t, theta, initial_lbd, current_best):
        nquery = 0
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
        return lbd_hi, nquery

    def cw_loss(self, logit, label, target=None):
        if target is not None:
            target = target.clone().cuda()
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
            _, argsort = logit.sort(dim=1, descending=True)
            target_is_max = argsort[:, 0].eq(target).long()
            second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
            target_logit = logit[torch.arange(logit.shape[0]), target]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return target_logit - second_max_logit
        else:
            label = label.clone().cuda()
            _, argsort = logit.sort(dim=1, descending=True)
            # print('True label:{}, The max label:{}, Second max label:{}'.format(label.item(), argsort[:, 0].item(), argsort[:, 1].item()))
            gt_is_max = argsort[:, 0].eq(label).long()
            second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
            gt_logit = logit[torch.arange(logit.shape[0]), label]
            second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
            return second_max_logit - gt_logit

    def g_function_bin_search(self, model, image, theta, initial_lbd, true_labels, target_labels, tol=1e-3):

        if target_labels is not None:
            start = initial_lbd + 1.0
            # assert start <= 100.0, "initial_lbd > 100 error! It is {}".format(start)
            target_label = model((image + start * theta).cuda()).max(1)[1].item()
            num_points = 400
            lmdb_list = torch.linspace(start, 0, num_points).view(-1, 1, 1, 1).to(image.device)
            perturbed = image + lmdb_list * theta
            pred_labels = model(perturbed.cuda()).max(1)[1]
            if pred_labels[0] != target_label:  # 处于决策边界附近，数值计算不稳定
                start *= 1.001  # 使其远离一点决策边界
                target_label = model((image + start * theta).cuda()).max(1)[1].item()
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
        if target_labels is None:
            lmdb_result, nquery = self.fine_grained_binary_search_local(model, image, true_labels, theta, initial_lbd,
                                                                        max_high_bound=max_high_bound, tol=tol, search_bound=False)
        else:
            lmdb_result, nquery = self.fine_grained_binary_search_local_targeted(model, image, target_labels, theta, initial_lbd,
                                                                        max_high_bound=max_high_bound, tol=tol, search_bound=False)

        if lmdb_result == float("inf"):
            log.warn("warn: float('inf') value of the distance along theta in get_grad of surrogate model!")
            lmdb_result = initial_lbd
        return lmdb_result, target_labels, nquery

    def get_g_grad(self, model, images, theta, initial_lbd, true_labels, target_labels):
        theta = theta.clone().detach()
        if images.size(-1) != model.input_size[-1]:
            images = F.interpolate(images, size=model.input_size[-1], mode='bilinear', align_corners=False)
            theta = F.interpolate(theta, size=model.input_size[-1], mode='bilinear', align_corners=False)
            theta /= self.norm(theta)
        with torch.no_grad():
            min_lmdb, target_labels, _ = self.g_function_bin_search(model, images.detach(), theta, initial_lbd, true_labels, target_labels)
        if target_labels is not None:
            target_labels = torch.tensor([target_labels]).long().cuda()

        with torch.enable_grad():
            theta.requires_grad_()
            loss = self.cw_loss(model(images + min_lmdb * theta / torch.norm(theta, p=2, dim=[1, 2, 3], keepdim=True)), true_labels, target_labels)
            grad_theta = torch.autograd.grad(-loss, theta, create_graph=False)[0]
        return grad_theta.detach()

    def get_prior_grad(self, images, theta, bound_radius, true_label, target_label):
        prior_grads = []
        if self.method == 'PARS':
            for surrogate_model in self.surrogate_models:
                prior_grad = self.get_g_grad(surrogate_model, images, theta, bound_radius, true_label, target_label)
                prior_grad = prior_grad / torch.norm(prior_grad, p=2, dim=[1, 2, 3], keepdim=True)
                prior_grads.append(prior_grad)
        return prior_grads

    def get_orthogonal_basis(self, images, theta, bound_radius, true_label, target_label):
        prior_grads = self.get_prior_grad(images, theta, bound_radius, true_label, target_label) if self.method == 'PARS' else {}

        noise_shape = (images.size(1), int(images.size(2)/self.dim_reduce_factor), int(images.size(3)/self.dim_reduce_factor)) if self.resized else None
        us = []
        for prior_grad in prior_grads:
            _prior_grad = F.interpolate(prior_grad, size=noise_shape[1:], mode='bilinear') if self.resized else prior_grad
            _prior_grad /= torch.norm(_prior_grad.view(-1), p=2, dim=0)
            us.append(_prior_grad.squeeze())
        for i in range(self.q - self.s):
            if self.resized:
                rv = torch.randn(noise_shape).cuda()
                # rv = F.interpolate(u, size=theta.size()[2:], mode='bilinear', align_corners=False)
            else:
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
        orthos_ = [u if not self.resized else F.interpolate(u.unsqueeze(0), size=theta.size()[2:], mode='bilinear', align_corners=False).squeeze() for u in orthos]
        orthos_ = torch.stack(orthos_, dim=0)
        if self.resized:
            orthos_ /= self.norm(orthos_)
        return orthos_

    def get_v_t_estimator(self, images, y, orthos, y_bound_radius, true_label, target_label, sigma=0.001):
        u_batch = []
        images_batch = []
        q_sign_estimator = self.q - self.s
        for orth in orthos:
            u = orth.unsqueeze(0)
            new_theta = y + sigma * u
            new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
            u_batch.append(u)
            images_batch.append(images + y_bound_radius * new_theta)
        images_batch = torch.cat(images_batch, 0)
        u_batch = torch.cat(u_batch, 0)  # B,C,H,W
        assert u_batch.dim() == 4
        sign = torch.ones(q_sign_estimator, device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(q_sign_estimator)], device='cuda').long()
            predict_labels = self.model(images_batch[self.s:]).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(q_sign_estimator)], device='cuda').long()
            predict_labels = self.model(images_batch[self.s:]).max(1)[1]
            sign[predict_labels != true_labels] = -1
        query = q_sign_estimator

        sign_grad = torch.sum(u_batch[self.s:] * sign.view(q_sign_estimator, 1, 1, 1), dim=0, keepdim=True)
        sign_grad = sign_grad / torch.sqrt(torch.sum(sign_grad * sign_grad))
        return sign_grad, query

    def get_zeta_t(self, prior_grads, L1, g_f_square, images, theta, bound_radius, true_label, target_label, sigma=0.001):
        df2_prior, query_conut = 0, 0
        for prior_grad in prior_grads:
            d_v, bs_count = self.differential_calculation(images, theta, bound_radius, prior_grad, true_label, target_label, sigma/500, sigma)
            df2_prior += d_v ** 2
            query_conut += bs_count
        Dt = df2_prior / g_f_square if self.method == 'PARS' else 0
        Dt = min(Dt, 0.6)
        zeta_pars = (Dt + (1-Dt) * (2/np.pi*(self.q-self.s-1)+1)/(self.d-self.s)) / (L1 * (Dt + (1-Dt) * (self.d-self.s)/(2/np.pi*(self.q-self.s-1)+1)))
        return zeta_pars, query_conut

    def differential_calculation(self, images, theta, bound_radius, theta_grad, true_label, target_label, tol, sigma=0.001):
        query_conut = 0
        sigma_ = sigma
        lbd = bound_radius
        for j in range(10):
            new_theta = theta + sigma_ * theta_grad
            new_theta /= self.norm(new_theta)
            max_high_bound = 100
            if lbd > max_high_bound:
                max_high_bound = lbd + 100
            if true_label is not None:
                perturb_bound_radius, bs_count = self.fine_grained_binary_search_local(
                    self.model, images, true_label, new_theta, lbd, max_high_bound, tol=tol, search_bound=False)
            else:
                perturb_bound_radius, bs_count = self.fine_grained_binary_search_local_targeted(
                    self.model, images, target_label, new_theta, lbd, max_high_bound, tol=tol, search_bound=False)

            query_conut += bs_count
            if perturb_bound_radius == float("inf") and j + 1 < 10:
                sigma_ = -sigma_ if sigma_ < 0 else -0.1 * sigma_
            elif perturb_bound_radius == lbd and j + 1 < 10:
                lbd *= 1.001
            else:
                if perturb_bound_radius == float('inf'):
                    perturb_bound_radius = bound_radius
                break

        return (perturb_bound_radius-bound_radius)/sigma_, query_conut

    def pars_grad(self, images, theta, orthos, initial_lbd, alpha, current_momentum, true_label, target_label=None, sigma=0.001):
        query = 0
        y = (1 - alpha) * theta + alpha * current_momentum
        y /= self.norm(y)

        prior_grad_binary_search_tol = sigma / 500
        if self.prior_grad_binary_search_tol is not None:
            prior_grad_binary_search_tol = self.prior_grad_binary_search_tol
        if true_label is not None:
            y_bound_radius, bs_count = self.fine_grained_binary_search_local(
                self.model, images, true_label, y, initial_lbd, initial_lbd + 100, prior_grad_binary_search_tol, search_bound=False)
            query += bs_count
        else:
            y_bound_radius, bs_count = self.fine_grained_binary_search_local_targeted(
                self.model, images, target_label, y, initial_lbd, initial_lbd + 100, prior_grad_binary_search_tol, search_bound=False)
            query += bs_count
        y_updata_is_inf = False
        if y_bound_radius == float("inf"):
            log.warn("warn: the returned boundary distance is float('inf') after the binary search for calculating the loss derivative.")
            y_updata_is_inf = True
            y, y_bound_radius = theta.detach(), initial_lbd

        # Get v_t
        v_t, es_query = self.get_v_t_estimator(images, y, orthos, y_bound_radius, true_label, target_label, sigma)
        query += es_query

        g1 = torch.zeros_like(theta)
        g2 = torch.zeros_like(theta)
        new_orthos = [v_t] + [prior_grad for prior_grad in orthos[:self.s]]

        differential_values = []
        for i, grad_theta_orth in enumerate(new_orthos):
            differential_value, bs_count = self.differential_calculation(images, y, y_bound_radius, grad_theta_orth, true_label,
                                                               target_label, prior_grad_binary_search_tol, sigma)
            query += bs_count
            differential_values.append(differential_value)

        if self.method == "ARS":
            g1 = differential_values[0] * new_orthos[0]
            g2 = 1/(1/self.d*(2/np.pi*(self.q-1)+1)) * g1
        elif self.method == "PARS":
            for idx, (grad_theta_orth, d_v) in enumerate(zip(new_orthos, differential_values)):
                if idx == 0:
                    coeff = (self.d-self.s)/(2/np.pi*(self.q-self.s-1)+1)
                else:
                    coeff = 1.0
                g1 += d_v * grad_theta_orth
                g2 += coeff * d_v * grad_theta_orth

        g_f_square = 0
        if self.method == 'PARS':
            df2_prior = 1e-10
            df2_vt = 0
            for i, d_v in enumerate(differential_values):
                dr2 = d_v ** 2
                if i == 0:
                    df2_vt += dr2
                else:
                    df2_prior += dr2

            g_f_square = df2_prior + (self.d-self.s)/(2/np.pi*(self.q-self.s-1)+1)*df2_vt
        return g1, g2, g_f_square, query, query-es_query, y_updata_is_inf, y_bound_radius

    def update_theta(self, xg, gg, g1, alpha, beta, images, label,
                     image_index, query, success_stop_queries, batch_image_positions):
        ls_count = 0
        if self.line_search:
            min_theta = xg  ## next theta
            min_g2 = gg  ## current g_theta
            for _ in range(15):
                # update theta by one step sgd
                new_theta = xg - alpha * g1
                new_theta /= torch.norm(new_theta.view(-1),p=2)
                tol = beta/500
                if self.tol is not None:
                    tol = self.tol
                if self.targeted:
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        self.model, images, label, new_theta, initial_lbd=min_g2, tol=tol)
                else:
                    new_g2, count = self.fine_grained_binary_search_local(
                        self.model, images, label, new_theta, initial_lbd=min_g2, tol=tol)
                ls_count += count
                query += count
                alpha = alpha * 2  # gradually increasing step size
                if new_g2 < min_g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                         success_stop_queries, batch_image_positions)
                else:
                    break
            if min_g2 >= gg:  ## if the above code failed for the init alpha, we then try to decrease alpha
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = xg - alpha * g1
                    new_theta /= torch.norm(new_theta)
                    tol = beta / 500
                    if self.tol is not None:
                        tol = self.tol
                    if self.targeted:
                        new_g2, count = self.fine_grained_binary_search_local_targeted(
                            self.model, images, label, new_theta, initial_lbd=min_g2, tol=tol)
                    else:
                        new_g2, count = self.fine_grained_binary_search_local(
                            self.model, images, label, new_theta, initial_lbd=min_g2, tol=tol)
                    ls_count += count
                    query += count
                    if new_g2 < gg:
                        min_theta = new_theta
                        min_g2 = new_g2
                        self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                             success_stop_queries, batch_image_positions)
                        break
            if alpha < 1e-4:  ## if the above two blocks of code failed
                alpha = 1.0
                log.info("{}-th image warns: not moving".format(image_index+1))
                beta = beta * 0.1
                beta = max(beta, 1e-8)

            return min_theta, min_g2, alpha, beta, query, ls_count
        else:
            while True:
                # update theta by one step sgd
                new_theta = xg - alpha * g1
                new_theta /= torch.norm(new_theta.view(-1), p=2)
                tol = beta/500
                if self.tol is not None:
                    tol = self.tol
                if self.targeted:
                    new_g2, count = self.fine_grained_binary_search_local_targeted(
                        self.model, images, label, new_theta, initial_lbd=gg, tol=tol)
                else:
                    new_g2, count = self.fine_grained_binary_search_local(
                        self.model, images, label, new_theta, initial_lbd=gg, tol=tol)
                ls_count += count
                query += count
                if new_g2 == float('inf') and alpha > 1e-4:
                    alpha *= 0.5
                else:
                    alpha = self.alpha
                    break

            if new_g2 == float('inf') or alpha <= 1e-4:
                log.info("warn: the returned boundary distance is float('inf') after the binary search.")
                return False
            if new_g2 < gg:
                min_theta = new_theta
                min_g2 = new_g2
                self.count_stop_query_and_distortion(images, images + min_theta * min_g2, query,
                                                     success_stop_queries, batch_image_positions)
            return new_theta, new_g2, alpha, beta, query, ls_count

    def untargeted_attack(self, image_index, images, true_labels):
        assert images.size(0) == 1

        alpha = self.alpha
        beta = self.beta
        L1 = 1.0 / alpha
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
            if self.model(images + theta).max(1)[1].item() != true_label:
                initial_lbd = torch.norm(theta.view(-1), p=2).item()
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(images, true_label, theta, initial_lbd, g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    log.info("{}-th image, {}-th iteration distortion: {:.4f}".format(image_index + 1, i, g_theta))
                    self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query, success_stop_queries,
                                                         batch_image_positions)
        ## fail if cannot find an adversarial direction within 200 Gaussian
        if g_theta == float('inf'):
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index + 1))
            return images, query, success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index + 1, g_theta, query[0].item()))
        #### Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        self.d = np.prod(images.size())
        self.s = len(self.surrogate_models)
        g_f_square = float("inf")
        gamma = L1
        current_momentum = xg.clone()
        for i in range(self.iterations):
            if query.min().item() >= self.maximum_queries:
                break
            # Get Orthogonal Basis u_i
            orthos = self.get_orthogonal_basis(images, xg, gg, true_labels, None)

            # Estimate_zeta_t
            zeta_pars, ls_count = self.get_zeta_t(orthos[:len(self.surrogate_models)], L1, g_f_square, images, xg, gg, true_label, None)
            query += ls_count
            ls_total += ls_count
            k = zeta_pars * gamma
            alpha_pars = (-k + np.sqrt(k * k + 4 * k)) / 2  # the gamma is converted to alpha_pars
            gamma = (1 - alpha_pars) * gamma

            ## gradient estimation
            g1, g2, g_f_square, grad_queries, ls_count, y_updata_is_inf, y_bound_radius = \
                self.pars_grad(images, xg, orthos, gg, alpha_pars, current_momentum, true_labels, None, beta)
            query += grad_queries
            ls_total += ls_count

            g1 = self.clip_grad_norm(g1, max_norm=self.clip_grad_max_norm)
            # g2 = self.clip_grad_norm(g2, max_norm=self.clip_grad_max_norm)
            if y_updata_is_inf:
                current_momentum = xg.clone()
            y = (1 - alpha_pars) * xg + alpha_pars * current_momentum
            y /= self.norm(y)

            ## Line search of the step size of gradient descent
            results = self.update_theta(
                y, y_bound_radius, g1, alpha, beta, images, true_label,
                image_index, query, success_stop_queries, batch_image_positions
            )
            if results == False:
                continue
            else:
                min_theta, min_g2, alpha, beta, query, ls_count = results

            current_momentum = current_momentum - zeta_pars / alpha_pars * g2
            current_momentum /= torch.norm(current_momentum.view(-1), p=2)

            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2

            ls_total += ls_count
            ## logging
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(image_index+1, i+1, gg, query[0].item()))


        if self.epsilon is None or gg <= self.epsilon:
            target = self.model(images + gg * xg).max(1)[1].item()
            log.info("{}-th image success distortion {:.4f} target {} queries {} LS queries {}".format(image_index+1, gg, target, query[0].item(), ls_total))
        # gg 是distortion
        distortion = torch.norm(gg * xg, p=2)
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        return images + gg * xg, query, success_stop_queries, torch.tensor([gg]).float(), torch.tensor([gg]).float() <= self.epsilon, xg


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
        L1 = 1.0 / alpha
        batch_image_positions = np.arange(image_index * self.batch_size,
                                          min((image_index + 1) * self.batch_size, self.total_images)).tolist()
        query = torch.zeros(images.size(0))
        success_stop_queries = query.clone()
        ls_total = 0

        num_samples = 1
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
                    xi = F.interpolate(xi, size=(self.model.input_size[-2], self.model.input_size[-1]), mode='bilinear', align_corners=False)
                xi = xi.cuda()
                yi_pred = self.model(xi).max(1)[1].item()
                query += 1
                if yi_pred != target_label:
                    continue

                theta = xi - images
                initial_lbd = torch.norm(theta.view(-1),p=2)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd,
                                                                      g_theta)
                query += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query, success_stop_queries, batch_image_positions)
                    log.info("{}-th image. Found initial target image with the distortion {:.4f}".format(image_index+1, g_theta))

                if i > 100:
                    break
        else:
            # xi = self.get_image_of_target_class(self.dataset, target_labels, self.model)
            xi = target_class_image
            theta = xi - images
            initial_lbd = torch.linalg.norm(theta).cpu()
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(images, target_label, theta, initial_lbd, g_theta)
            query += count
            best_theta, g_theta = theta, lbd
            self.count_stop_query_and_distortion(images, images + best_theta * g_theta, query, success_stop_queries, batch_image_positions)
        if g_theta == np.inf:
            log.info("{}-th image couldn't find valid initial, failed!".format(image_index + 1))
            return images, query, success_stop_queries, torch.zeros(images.size(0)), torch.zeros(images.size(0)), best_theta
        log.info("{}-th image found best distortion {:.4f} using {} queries".format(image_index + 1, g_theta, query[0].item()))
        #### Begin Gradient Descent.
        xg, gg = best_theta, g_theta
        self.d = np.prod(images.size())
        self.s = len(self.surrogate_models)
        g_f_square = float("inf")
        gamma = L1
        current_momentum = xg.clone()
        for i in range(self.iterations):
            if query.min().item() >= self.maximum_queries:
                break
            # Get Orthogonal Basis u_i
            orthos = self.get_orthogonal_basis(images, xg, gg, None, target_labels)
            # prior_grads = self.get_prior_grad(images, xg, gg, None, target_label)

            # get_zeta_t
            zeta_pars, ls_count = self.get_zeta_t(orthos[:len(self.surrogate_models)], L1, g_f_square, images, xg, gg, None, target_labels)
            query += ls_count
            ls_total += ls_count
            k = zeta_pars * gamma
            alpha_pars = (-k + np.sqrt(k * k + 4 * k)) / 2  # the gamma is converted to alpha_pars
            gamma = (1 - alpha_pars) * gamma

            ## gradient estimation
            g1, g2, g_f_square, grad_queries, ls_count, y_updata_is_inf, y_bound_radius = \
                self.pars_grad(images, xg, orthos, gg, alpha_pars, current_momentum, None, target_labels, beta)
            query += grad_queries
            ls_total += ls_count

            g1 = self.clip_grad_norm(g1, max_norm=self.clip_grad_max_norm)
            # g2 = self.clip_grad_norm(g2, max_norm=self.clip_grad_max_norm)
            if y_updata_is_inf:
                current_momentum = xg.clone()
            y = (1 - alpha_pars) * xg + alpha_pars * current_momentum
            y /= self.norm(y)

            ## Line search of the step size of gradient descent
            results = self.update_theta(
                y, y_bound_radius, g1, alpha, beta, images, target_label,
                image_index, query, success_stop_queries, batch_image_positions
            )
            if results == False:
                continue
            else:
                min_theta, min_g2, alpha, beta, query, ls_count = results

            current_momentum = current_momentum - zeta_pars / alpha_pars * g2
            current_momentum /= torch.norm(current_momentum.view(-1), p=2)

            ## if all attemps failed, min_theta, min_g2 will be the current theta (i.e. not moving)
            xg, gg = min_theta, min_g2

            ls_total += ls_count
            ## logging
            log.info("{}-th Image, iteration {}, distortion {:.4f}, num_queries {}".format(
                image_index + 1, i + 1, gg, query[0].item()))

        log.info(
            "{}-th image success distortion {:.4f} queries {} stop queries {}".format(
                image_index + 1, gg, query[0].item(), success_stop_queries[0].item()))
        adv_target = self.model(images + gg * xg).max(1)[1].item()
        if adv_target == target_label:
            log.info("{}-th image attack successfully! Distortion {:.4f} target {} queries:{} success stop queries:{} LS queries:{}".format(
                image_index + 1, gg, adv_target, query[0].item(), success_stop_queries[0].item(), ls_total))
        else:
            log.info("{}-th image is failed to find targeted adversarial example.".format(image_index+1))

        distortion = torch.norm(gg * xg, p=2)
        assert distortion.item() - gg < 1e-4, "gg:{:.4f}  dist:{:.4f}".format(gg, distortion.item())
        # success_stop_queries = torch.clamp(success_stop_queries, 0, self.maximum_queries)
        return images + gg * xg, query, success_stop_queries, torch.tensor([gg]).float(), torch.tensor(
            [gg]).float() <= self.epsilon, xg


    def count_stop_query_and_distortion(self, images, perturbed, query, success_stop_queries,
                                        batch_image_positions):
        dist = torch.norm((perturbed - images).view(images.size(0), -1), p=2, dim=1)
        if torch.sum(dist > self.epsilon).item() > 0:
            working_ind = torch.nonzero(dist > self.epsilon).view(-1).cpu()
            success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(batch_image_positions):
            self.distortion_all[index_over_all_images][query[inside_batch_index].item()] = dist[inside_batch_index].item()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def attack_all_images(self, args, arch_name, result_dump_path):
        if args.targeted and args.target_type == "load_random":
            loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
            loaded_target_labels = torch.from_numpy(loaded_target_labels).long()

        for batch_index, (images, true_labels) in enumerate(self.dataset_loader):
            self.set_seed(args.seed)
            if batch_index < args.start-1:
                continue
            if batch_index >= args.total_images:
                continue
            if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                images = F.interpolate(images, size=(self.model.input_size[-2], self.model.input_size[-1]),
                                       mode='bilinear', align_corners=False)
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
                correct_surrogate = pred_surrogate.eq(true_labels.cuda()).int()  # shape = (batch_size,)
                if correct_surrogate.item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                    log.info("{}-th original image is classified incorrectly by surrogate model, skip!".format(batch_index + 1))
                    all_surrogate_correct = False
                    break
            if not all_surrogate_correct:
                continue

            # PARS logic
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

            if args.targeted:
                # target_class_image = self.get_image_of_target_class(self.dataset, target_labels, self.model)
                target_class_image = select_random_image_of_target_class(self.dataset, target_labels, self.model, args.load_random_class_image)
                if target_class_image is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
                target_class_image = target_class_image.cuda()
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta = \
                    self.targeted_attack(batch_index, images, target_labels, target_class_image)
            else:
                adv_images, query, success_query, distortion_with_max_queries, success_epsilon, theta = \
                    self.untargeted_attack(batch_index, images, true_labels)
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
