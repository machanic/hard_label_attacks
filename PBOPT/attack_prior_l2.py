import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict, defaultdict
import argparse
import json
import os
import os.path as osp
import random
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel
import glog as log
import numpy as np
import torch
from torch.nn import functional as F
from dataset.dataset_loader_maker import DataLoaderMaker
from PBOPT.bo_l2 import BayesOpt

def negative_cw_loss(logits, targets, is_targeted=False, num_classes=1000):
    if args.dataset == "ImageNet":
        num_classes = 1000
    elif  args.dataset == "CIFAR-10":
        num_classes = 10
    onehot_targets = torch.zeros([targets.size(0), num_classes]).to(targets.device)
    onehot_targets[torch.arange(targets.size(0), device=targets.device), targets] = 1.0

    target_logits = torch.sum(onehot_targets * logits, dim=1)
    other_logits = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

    if is_targeted:
        loss = other_logits - target_logits
    else:
        loss = target_logits - other_logits

    return loss

class Objective(object):
    def __init__(self, model, idx, data, target, target_data=None, original_size=32,
                 reduce_size=None, is_targeted=False, nograd=True, mode='bilinear', init_q=0):
        self.model = model.to(data.device)
        self.original_size = original_size
        self.reduce_size = reduce_size if reduce_size is not None else original_size
        self.is_targeted = is_targeted
        self.idx = idx
        self.data = data
        self.target_data = target_data
        self.target = target
        self.nograd = nograd
        self.mode = mode
        self.init_q = init_q
        self.bound_th = 0.1
        self.th = 0.1
        self.bound_th_s = 0.1
        self.th_s = 0.1

    def get_logits(self, pert):
        image_pert = pert.reshape([-1, 3, self.reduce_size, self.reduce_size])
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
            elif self.mode == 'nearest':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise
        logits = self.model(torch.clamp(self.data + image_pert, min=0., max=1.))
        return logits

    def _is_adversarial(self, x):
        output = self.model(torch.clamp(x, min=0., max=1.)).max(1)[1]
        if self.is_targeted:
            return output == self.target
        else:
            return output != self.target

    def fine_grained_binary_search(self, x0, theta, initial_lbd=1.0):
        lbd = initial_lbd
        bound_th = self.bound_th + 1
        query = 1
        if not self._is_adversarial(x0 + lbd * theta):
            lbd_lo = lbd
            lbd_hi = lbd * bound_th
            query += 1
            while not self._is_adversarial(x0 + lbd_hi * theta):
                lbd_hi = lbd_hi * bound_th
                query += 1
                if query == 20:
                    return torch.tensor(1e5), query - 1
        else:
            lbd_hi = lbd
            lbd_lo = lbd * (1 / bound_th)
            query += 1
            while self._is_adversarial(x0 + lbd_lo * theta):
                lbd_lo = lbd_lo * (1 / bound_th)
                query += 1

        tot_count = 0
        old_lbd_mid = lbd_hi
        while (lbd_hi - lbd_lo) > self.th * initial_lbd:
            tot_count += 1
            lbd_mid = (lbd_lo + lbd_hi) / 2.0
            query += 1
            if self._is_adversarial(x0 + lbd_mid * theta):
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
            if old_lbd_mid == lbd_mid or tot_count > 200:
                break
            old_lbd_mid = lbd_mid

        self.bound_th = max(self.bound_th*0.999, 0.02)
        self.th = max(self.th*0.997, 0.002)
        # self.th = self.bound_th
        return torch.tensor(lbd_hi), query

    def __call__(self, pert, ldb):
        image_pert = pert.reshape([-1, 3, self.reduce_size, self.reduce_size])
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='bilinear', align_corners=True) # fixme
            elif self.mode == 'nearest':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise
        res, nquery = [], 0
        for i in range(len(image_pert)):
            theta = image_pert[i]/torch.norm(image_pert[i], p=2)
            _res, query = self.fine_grained_binary_search(self.data, theta, ldb)
            distorition = torch.norm(torch.clamp(self.data + _res * theta, min=0., max=1.) - self.data, p=2)
            res.append(distorition)
            nquery += query
        return -torch.stack(res), nquery

class DistortionWithCWGrad(torch.autograd.Function):

    @staticmethod
    def set_model(model):  # 通过静态方式设置
        DistortionWithCWGrad.model = model

    @staticmethod
    def forward(ctx, x0, theta, ldb, target, is_targeted, fine_grained_search_func):
        # theta shape = (B,C,H,W)
        norms = theta.norm(p=2, dim=(1, 2, 3), keepdim=True)
        norms = norms.clamp_min(1e-12)
        theta = theta / norms
        with torch.no_grad():
            l2_distance = fine_grained_search_func(x0, theta, ldb)  # shape = (B,)
        ctx.save_for_backward(x0, theta, l2_distance)
        ctx.is_targeted = is_targeted
        ctx.target = target
        return l2_distance  # (B,)；B=1 时就是形状 (1,)

    @staticmethod
    def backward(ctx, grad_output):
        x0, theta, l2_distance = ctx.saved_tensors
        B = theta.size(0)
        l2_distance = l2_distance.view(B,1,1,1)
        finite_mask = torch.isfinite(l2_distance.view(-1))
        idx = torch.nonzero(finite_mask, as_tuple=False).view(-1)
        grad_theta_full = torch.zeros_like(theta)
        if idx.numel() == 0:
            return None, grad_output.view(B, 1, 1, 1) * grad_theta_full, None, None, None, None, None
        with torch.enable_grad():
            x0 = x0.detach()
            theta_sel = theta.index_select(0, idx).detach().requires_grad_(True)
            lambda_sel = l2_distance.index_select(0, idx).detach().requires_grad_(True)
            x_adv_sel = torch.clamp(x0 + lambda_sel * theta_sel/torch.norm(theta_sel,p=2,dim=tuple(range(1, theta_sel.ndim)),keepdim=True),0,1.0)   # 有可能传入无穷大的l2_distance
            logits = DistortionWithCWGrad.model(x_adv_sel)
            loss = negative_cw_loss(logits, ctx.target, ctx.is_targeted)
            # grad_theta = torch.autograd.grad(-loss, theta, create_graph=False, grad_outputs=torch.ones_like(loss))[0]
            grad_h_theta_sel, grad_lambda_sel = torch.autograd.grad(
                outputs=loss,
                inputs=(theta_sel,lambda_sel),
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
                allow_unused=False
            )
            eps = 1e-12
            grad_g_func_theta = -grad_h_theta_sel / (grad_lambda_sel.view(-1, 1, 1, 1) + eps)
            grad_theta_full.index_copy_(0, idx, grad_g_func_theta)
        del logits, loss, x_adv_sel, grad_h_theta_sel, grad_lambda_sel, grad_g_func_theta
        return None, grad_output.view(B, 1, 1, 1) * grad_theta_full, None, None, None, None, None

class SurrogateObjective(Objective):
    def __init__(self, model, idx, data, target, target_data=None, original_size=32,
                 reduce_size=None, is_targeted=False, nograd=True, mode='bilinear', init_q=0):
        super(SurrogateObjective, self).__init__(model,idx,data,target,target_data,original_size,reduce_size,is_targeted,nograd,
                                                 mode,init_q)
        DistortionWithCWGrad.set_model(model)

    def parallel_find_boundary(self, x0, theta, initial_ldb=1.0):
        # shape of x0 = 1,C,H,W
        # shape of theta = B,C,H,W, eg., 20,3,224,224
        num_points = 10
        assert x0.dim() == 4, 'x0 should be a 4D tensor'
        assert theta.dim() == 4, 'theta should be a 4D tensor'
        x0_u = x0.unsqueeze(1)  # shape = (1, 1, C, H, W)
        theta_u = theta.unsqueeze(1)  # shape = (B, 1, C, H, W)
        if not self.is_targeted:
            # assert self.model(x0).max(1)[1] == self.target, "initial_ldb is incorrect, because it cannot be classified as true label for x0"
            lmdb_min = 0
            lmbd_max = 7 * initial_ldb
            B, C, H, W = theta.shape
            orig_B = B
            while True:
                lmbd_list = torch.linspace(lmdb_min, lmbd_max, num_points, device=x0.device).view(1, num_points, 1, 1, 1)
                perturbed = torch.clamp(x0_u + lmbd_list * theta_u,min=0,max=1.0) # (1,1,C,H,W) + (1,num_points,1,1,1) * (B,1,C,H,W) = (B, num_points, C, H, W)
                perturbed = perturbed.view(B * num_points, C, H, W)
                pred_labels = self.model(perturbed.cuda()).max(1)[1]  # the first time batched query, torch.Size([B * num_points])
                pred_labels = pred_labels.view(B, num_points)
                any_non_true = pred_labels != self.target  # [0,0,0,0,0,1,1,1,1], [0,0,1,1,1,1,1,1], shape = (B, num_points)
                non_all_failed_mask = any_non_true.any(dim=1)  # shape = (B,)，至少有一个True
                all_rows_all_failed = (~non_all_failed_mask).all().item()
                all_rows_have_success = non_all_failed_mask.all().item()
                if not all_rows_all_failed:
                    break
                lmbd_max *= 2

            success_attack_indices = torch.arange(B, device=x0.device)
            if not all_rows_have_success:
                success_attack_indices = torch.nonzero(non_all_failed_mask, as_tuple=False).view(-1) # 拿到这些行的index
                theta = theta[success_attack_indices]
                B = success_attack_indices.size(0)
                any_non_true = any_non_true[success_attack_indices]
                theta_u = theta_u[success_attack_indices]

            idx = torch.arange(num_points, device=any_non_true.device).unsqueeze(0).expand(B, num_points) # 构造每列的下标 [0,1,2,...,num_points-1] 并广播到 (B, num_points)
            mask = torch.where(any_non_true, idx, torch.full_like(idx, 1000000)) # 把 False 的位置替换成 1000000（一定比任何合法下标都大）
            index_of_first_not_true_label = mask.min(dim=1).values # 1D tensor with shape = (B, )

            reversed_mask = torch.where(~any_non_true, idx, torch.full_like(idx, -1)) # [0,1,2,3,4,-1,-1,-1,-1], [0,1,-1,-1,-1,-1,-1,-1], shape = (B, num_points)
            index_of_last_true_label = reversed_mask.max(dim=1).values  # 1D tensor with shape = (B, )
            lmbd_exp = lmbd_list.repeat(B,1,1,1,1) # (B,num_points,1,1,1)

            b_idx = torch.arange(B, device=lmbd_exp.device)
            start_position_list = lmbd_exp[b_idx, index_of_last_true_label].view(-1) # shape = (B, )
            end_position_list = lmbd_exp[b_idx, index_of_first_not_true_label].view(-1) # shape = (B,)
            t = torch.linspace(0, 1, num_points, device=start_position_list.device)  # (num_points,)
            starts = start_position_list.view(B, 1)
            ends = end_position_list.view(B, 1)
            while True:
                fine = starts + (ends - starts) * t.view(1, num_points)
                lmbd_list_fine_grained = fine.view(B, num_points, 1, 1, 1)

                perturbed_fine_grained = torch.clamp(x0_u + lmbd_list_fine_grained * theta_u, min=0, max=1.0) # (1,1,C,H,W) + (B,num_points,1,1,1) * (B,1,C,H,W) = (B, num_points, C, H, W)
                perturbed_fine_grained = perturbed_fine_grained.view(B * num_points, C, H, W)
                pred_labels = self.model(perturbed_fine_grained).max(1)[1]  # the second time batched query
                pred_labels = pred_labels.view(B, num_points)
                any_non_true_fine_grained = pred_labels != self.target
                row_any = any_non_true_fine_grained.any(dim=1)
                all_rows_have_true = row_any.all().item()
                if all_rows_have_true:
                    break
                ends *= 1.01
            mask_fine_grained = torch.where(any_non_true_fine_grained, idx, torch.full_like(idx, 1000000))  # 把 False 的位置替换成 1000000（一定比任何合法下标都大）
            index_of_first_not_true_label_fine_grained = mask_fine_grained.min(dim=1).values  # 1D tensor with shape = (B, )
            final_ldb = lmbd_list_fine_grained[b_idx, index_of_first_not_true_label_fine_grained].view(-1)  # shape = (B,)
            final_ldb_full = torch.full((orig_B,), float("inf"), device=final_ldb.device)
            final_ldb_full.index_copy_(0, success_attack_indices, final_ldb)
            return final_ldb_full

        else:
            raise NotImplementedError("Targeted attacks are not implemented yet!")



    def __call__(self, theta, ldb, use_parallel_search=True):
        DistortionWithCWGrad.model.eval()
        theta = theta.view([-1, 3, self.reduce_size, self.reduce_size])
        B = theta.size(0)
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                theta = torch.nn.functional.interpolate(theta, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
            elif self.mode == 'nearest':
                theta = torch.nn.functional.interpolate(theta, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise

        l2_distance = DistortionWithCWGrad.apply(
            self.data,
            theta,
            ldb,  # 当前的扰动幅度
            self.target,  # 目标标签（仅当目标攻击时）
            self.is_targeted,
            self.parallel_find_boundary,
        )
        # finite_mask = torch.isfinite(l2_distance)
        # lambda_safe = torch.where(finite_mask, l2_distance, torch.zeros_like(l2_distance))
        # adv_x = torch.clamp(self.data + lambda_safe.view(B, 1, 1, 1) * theta / torch.norm(theta, dim=tuple(range(1, theta.ndim)), keepdim=True), 0., 1.)
        # distortion = torch.norm(adv_x - self.data, p=2, dim=(1, 2, 3)).view(-1)
        # distortion = torch.where(finite_mask, distortion, torch.full_like(distortion, float('inf')))
        # return -distortion  # (B,)；B=1 时就是形状 (1,)

        finite_mask = torch.isfinite(l2_distance.view(-1))
        idx = torch.nonzero(finite_mask, as_tuple=False).view(-1)  # (M,)
        distortion_full = torch.full((B,), float("inf"), device=l2_distance.device, dtype=self.data.dtype)
        if idx.numel() > 0:
            lam_sel = l2_distance.index_select(0, idx).view(-1, 1, 1, 1)  # (M,1,1,1)
            theta_sel = theta.index_select(0, idx)  # (M,C,H,W)
            theta_sel = theta_sel / torch.norm(theta_sel, p=2, dim=(1,2,3), keepdim=True)
            adv_x_sel = torch.clamp(self.data + lam_sel * theta_sel, 0., 1.)
            distortion_sel = torch.norm(adv_x_sel - self.data, p=2, dim=tuple(range(1, adv_x_sel.ndim)))  # (M,)
            distortion_full = distortion_full.scatter(0, idx, distortion_sel)
        return -distortion_full  # (B,)；B=1 时就是形状 (1,)

class PriorBayesianOPT(object):
    def __init__(self, args, model, surrogate_model, total_images):
        self.model = model
        self.surrogate_model = surrogate_model
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size, model.arch)
        self.total_images = total_images    # len(self.dataset_loader.dataset)
        self.batch_size = 1

        self.distortion = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros(self.total_images)  # number of images

    def count_query_and_distortion(self, dist, query):
        q = int(query.item()) if torch.is_tensor(query) else int(query)
        for inside_batch_index, index_over_all_images in enumerate(self.batch_image_positions):
            self.distortion[index_over_all_images][q] = dist

    def run_bo(self, obj, bounds, maximum_queries, device='cuda', prior_obj=None, scale='adapt'):
        self.batch_image_positions = np.arange(obj.idx * self.batch_size, min((obj.idx + 1)*self.batch_size, self.total_images)).tolist()
        bo = BayesOpt(obj, bounds, device=device, n_opt=1, n_init=10, n_past=10,
                      normalize_y=True, exact_fval=True, prior_f=prior_obj, scale=scale)
        if scale != 'fixed_only':
            bo.initialize(bo.gen_rand_init() if not obj.is_targeted else obj.target_data)
        else:
            bo.initialize(obj.target_data if obj.is_targeted else None)
        self.count_query_and_distortion(abs(bo.f.get_opt()), bo.f.call_count)
        x, _, _ = bo.run(maximum_queries=maximum_queries, freq_fit_hyper=10, save_dist=self.count_query_and_distortion)

        log.info('Total queries: {}'.format(bo.f.call_count.item()))
        return x

    def attack_all_images(self, args, tmp_result_path, result_json_path):
        reduce_size = args.size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if osp.exists(tmp_result_path):
            with open(tmp_result_path, "r") as file_obj:
                json_content = json.load(file_obj)
                args.start = int(json_content["batch_idx"])+2
                for key in ['correct_all', 'distortion']:
                    if key in json_content:
                        setattr(self, key, torch.from_numpy(np.asarray(json_content[key])).float())

        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            set_seed(args.seed)
            if batch_idx < args.start-1:
                continue
            if batch_idx >= args.total_images:
                continue
            if images.size(-1) != self.model.input_size[-1]:
                images = F.interpolate(images, size=self.model.input_size[-1], mode='bilinear', align_corners=False)
            selected = torch.arange(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, self.total_images))
            data, true_label = images.to(device=device), true_labels.to(device)

            assert data.shape[0] == 1 and true_label.shape[0] == 1
            with torch.no_grad():
                logit = self.model(data)
                pred = logit.argmax(dim=1)
                correct = pred.eq(true_label).float()

                if not correct.item():
                    log.info(f'Image {batch_idx+1} is classified incorrectly!')
                    continue

            if self.surrogate_model is not None:
                with torch.no_grad():
                    logit_surrogate = self.surrogate_model(data)
                    pred_surrogate = logit_surrogate.argmax(dim=1).to(device=device)
                    correct_surrogate = pred_surrogate.eq(true_label).int()  # shape = (batch_size,)
                    if correct_surrogate.item() == 0:  # we must skip any image that is classified incorrectly before attacking, otherwise this will cause infinity loop in later procedure
                        log.info("{}-th original image is classified incorrectly by surrogate model {}, skip!".format(batch_idx + 1, args.surrogate_arch))
                        continue

            dim = data.shape[0] * data.shape[1] * reduce_size * reduce_size  # the dimension is the dimension of the reduced size
            bounds = [[-1, 1]] * dim

            if not args.targeted:
                log.info(f'Attack image {batch_idx+1}, original label {true_label.item()}')
                obj = Objective(self.model, batch_idx, data, true_label, original_size=self.model.input_size[-1],
                                reduce_size=reduce_size, is_targeted=args.targeted, mode=args.dr, nograd=True)
                prior_obj = SurrogateObjective(self.surrogate_model, batch_idx, data, true_label,
                                               original_size=self.model.input_size[-1], reduce_size=reduce_size,
                                               is_targeted=args.targeted, mode=args.dr) if self.surrogate_model else None
                image_pert = self.run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)

            else:
                raise NotImplementedError("The targeted attack is not implemented.")
            # else:
            #     assert true_label.shape[0] == 1
            #     if args.target_type == 'random':
            #         target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
            #                                       size=true_labels.size()).long().cuda()
            #         invalid_target_index = target_labels.eq(true_labels)
            #         while invalid_target_index.sum().item() > 0:
            #             target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
            #                       size=target_labels[invalid_target_index].shape).long().cuda()
            #             invalid_target_index = target_labels.eq(true_labels)
            #     elif args.target_type == 'least_likely':
            #         logits = self.model(images)
            #         target_labels = logits.argmin(dim=1).to(device)
            #     elif args.target_type == "increment":
            #         target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset]).to(device)
            #     else:
            #         raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))
            #
            #     if args.dataset == "ImageNet":
            #         val_dataset = ImageNetDataset(IMAGE_DATA_ROOT[args.dataset], target_labels.item(), "validation")
            #     elif args.dataset == "CIFAR-10":
            #         val_dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[args.dataset], target_labels, "validation")
            #     val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)
            #     target_data, query = list(), 0
            #     init_num_x = 10
            #     for i, (xi, yi) in enumerate(val_dataset_loader):
            #         if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
            #             xi = F.interpolate(xi, size=(self.model.input_size[-2], self.model.input_size[-1]),
            #                                mode='bilinear', align_corners=False)
            #         xi = xi.to(device)
            #         yi_pred = self.model(xi).max(1)[1]
            #         query += 1
            #         if yi_pred != target_labels:
            #             continue
            #         theta = xi - data
            #         theta = F.interpolate(theta, size=(reduce_size, reduce_size), mode='bilinear', align_corners=False)
            #         target_data.append(theta.view(1, -1)/torch.norm(theta.view(-1), p=2))
            #         if len(target_data) == init_num_x:
            #             break
            #     if len(target_data) == 0:
            #         print('can‘t find target_data, skip!')
            #         continue
            #     target_data = torch.cat(target_data, dim=0)
            #     # print(target_data.shape)
            #
            #     log.info(f'Attack image {batch_idx+1}, original label {true_label.item()}, new label {target_labels}')
            #     obj = Objective(self.model, batch_idx, data, target_labels, target_data, original_size=self.model.input_size[-1], reduce_size=reduce_size,
            #                     is_targeted=args.targeted, mode=args.dr, nograd=True, init_q=query)
            #     prior_obj = SurrogateObjective(self.surrogate_model, batch_idx, data, true_label,
            #                                    original_size=self.model.input_size[-1], reduce_size=reduce_size,
            #                                    is_targeted=args.targeted, mode=args.dr) if self.surrogate_model else None
            #     image_pert, query = \
            #         self.run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)

            self.correct_all[selected] = correct.detach().float().cpu()
            meta_info_dict = {"avg_correct": self.correct_all.sum().item()/args.total_images,
                              "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                              "distortion": self.distortion,
                              "batch_idx" : batch_idx,
                              "args": vars(args)}
            with open(tmp_result_path, 'w') as tmp_result_file_obj:
                json.dump(meta_info_dict, tmp_result_file_obj, sort_keys=True)


        log.info('{} is attacked finished ({} images)'.format(args.arch, self.total_images))
        log.info('Saving results to {}'.format(result_json_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "distortion": self.distortion,
                          "args": vars(args)}
        with open(result_json_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write stats info to {}".format(result_json_path))


def get_exp_dir_name(dataset, surrogate_arch, norm, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.ablation_study:
        if surrogate_arch is not None:
            dirname = 'PriorBayesianOPT-{}-{}-{}/ablation_study'.format(dataset, norm, target_str)
        else:
            dirname = 'BayesianOPT-{}-{}-{}/ablation_study'.format(dataset, norm, target_str)
        return dirname

    if surrogate_arch is not None:
        if args.attack_defense:
            dirname = 'PriorBayesianOPT_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            dirname = 'PriorBayesianOPT-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'BayesianOPT_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            dirname = 'BayesianOPT-{}-{}-{}'.format(dataset, norm, target_str)
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def set_seed(seed):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--max-queries', type=int, default=2000)
    parser.add_argument('--norm', type=str, default='l2', choices=["l2"], required=True, help='Which lp constraint to run bandits [linf|l2]')
    # parser.add_argument('--json-config', type=str, default='configures/PBOPT.json',
    #                     help='a configures file to be passed in instead of arguments')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for bo attack.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-arch', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-defense-model', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-defense-eps', default=None, type=str, help='network architecture')
    parser.add_argument('--surrogate-defense-norm', default=None, type=str, help='network architecture')
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target-type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--attack-defense', action="store_true")
    parser.add_argument('--defense-model', type=str, default=None)
    parser.add_argument('--defense-norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense-eps', type=str, default="")
    parser.add_argument('--load-random-class-image', action='store_true', help='load a random image from the target class')
    parser.add_argument('--bo-scale', default='adapt', type=str, choices=['fixed', 'adapt', 'fixed_only'],
                        help='how to use surrogate models to guide optimization, see the doc in bo.py')
    parser.add_argument('--size', default=56, type=int, help='default=32: not using dimension reduction.'
                        'If <32, then reduce dimension to size*size*3')
    parser.add_argument('--start', type=int, default=0, help='skipping the first `start` images')
    parser.add_argument('--dr', type=str, default='bilinear', choices=['bilinear', 'nearest'], help='the dimension reduction algorithm')
    parser.add_argument('--total-images', type=int, default=1000)
    parser.add_argument('--ablation-study',action="store_true")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print("using GPU {}".format(args.gpu))

    if args.attack_defense:
        if args.surrogate_arch is not None:
            if args.dataset == "ImageNet":
                defense_surrogate_arch_str = "{surrogate_arch}({surrogate_defense_model}_{surrogate_defense_norm}_{surrogate_defense_eps})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model="AT" if args.surrogate_defense_model.startswith(
                        "adv_train") else args.surrogate_defense_model,
                    surrogate_defense_norm=args.surrogate_defense_norm,
                    surrogate_defense_eps=args.surrogate_defense_eps)
            else:
                defense_surrogate_arch_str = "{surrogate_arch}({surrogate_defense_model})".format(
                    surrogate_arch=args.surrogate_arch,
                    surrogate_defense_model=args.surrogate_defense_model)
            if args.dataset == "ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps})_surrogate_{surrogate_arch}.log".format(
                                             arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith(
                                                 "adv_train") else args.defense_model,
                                             arch_defense_norm=args.defense_norm,
                                             arch_defense_eps=args.defense_eps,
                                             surrogate_arch=defense_surrogate_arch_str))
                save_result_path = osp.join(args.exp_dir,
                                            "/{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps})_surrogate_{surrogate_arch}.json".format(
                                                arch=args.arch,
                                                arch_defense_model="AT" if args.defense_model.startswith(
                                                    "adv_train") else args.defense_model,
                                                arch_defense_norm=args.defense_norm,
                                                arch_defense_eps=args.defense_eps,
                                                surrogate_arch=defense_surrogate_arch_str))

            else:
                log_file_path = osp.join(args.exp_dir,
                                         "run_{arch}({arch_defense_model})_surrogate_{surrogate_arch}.log".format(
                                             arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith(
                                                 "adv_train") else args.defense_model,
                                             surrogate_arch=defense_surrogate_arch_str))
                save_result_path = osp.join(args.exp_dir,
                                            "/{arch}({arch_defense_model})_surrogate_{surrogate_arch}.json".format(
                                                arch=args.arch,
                                                arch_defense_model="AT" if args.defense_model.startswith(
                                                    "adv_train") else args.defense_model,
                                                surrogate_arch=defense_surrogate_arch_str))
        else:
            if args.dataset == "ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps}).log".format(
                                             arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith(
                                                 "adv_train") else args.defense_model,
                                             arch_defense_norm=args.defense_norm,
                                             arch_defense_eps=args.defense_eps))
                save_result_path = osp.join(args.exp_dir,
                                            "/{arch}({arch_defense_model}_{arch_defense_norm}_{arch_defense_eps}).json".format(
                                                arch=args.arch,
                                                arch_defense_model="AT" if args.defense_model.startswith(
                                                    "adv_train") else args.defense_model,
                                                arch_defense_norm=args.defense_norm,
                                                arch_defense_eps=args.defense_eps))

            else:
                log_file_path = osp.join(args.exp_dir, "run_{arch}({arch_defense_model}).log".format(
                    arch=args.arch, arch_defense_model="AT" if args.defense_model.startswith(
                        "adv_train") else args.defense_model))
                save_result_path = osp.join(args.exp_dir, "/{arch}({arch_defense_model}).json".format(
                    arch=args.arch,
                    arch_defense_model="AT" if args.defense_model.startswith("adv_train") else args.defense_model))
    else:
        if args.surrogate_arch is not None:
            log_file_path = osp.join(args.exp_dir, "run_{}_surrogate_{}.log".format(args.arch, args.surrogate_arch))
            save_result_path = osp.join(args.exp_dir, "{}_surrogate_{}.json".format(args.arch, args.surrogate_arch))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch))
            save_result_path = osp.join(args.exp_dir, '{}.json'.format(args.arch))
    p = Path(save_result_path)
    tmp_result_path = str(p.parent / ("tmp_" + p.name))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None



    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)
    log.info("Begin attack {} on {}, result will be saved to {}".format(args.arch, args.dataset, save_result_path))
    if args.attack_defense:
        model = DefensiveModel(args.dataset, args.arch, no_grad=True, defense_model=args.defense_model,eps =args.defense_eps,norm= args.defense_norm)
    else:
        model = StandardModel(args.dataset, args.arch, no_grad=True)
    model.cuda()
    model.eval()
    surrogate_model = None
    if args.surrogate_arch is not None:
        if args.attack_defense:
            surrogate_model = DefensiveModel(args.dataset, args.surrogate_arch, no_grad=False,
                                             defense_model=args.surrogate_defense_model,eps =args.surrogate_defense_eps,
                                             norm=args.surrogate_defense_norm)
        else:
            surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False)
        surrogate_model.cuda()
        surrogate_model.eval()

    attacker = PriorBayesianOPT(args, model, surrogate_model, total_images=args.total_images)
    attacker.attack_all_images(args, tmp_result_path, save_result_path)
    os.remove(tmp_result_path)
    model.cpu()
    if args.surrogate_arch is not None:
       surrogate_model.cpu()
