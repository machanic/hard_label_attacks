import glob
import sys
import os
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict, defaultdict
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
from config import IMAGE_DATA_ROOT, CLASS_NUM, PROJECT_PATH
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR10Dataset
from PBO_OPT.bo_l2 import BayesOpt


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
        self.th = 0.01

    def _is_adversarial(self, x):
        output = self.model(torch.clamp(x, min=0., max=1.)).max(1)[1]
        if self.is_targeted:
            return output == self.target
        else:
            return output != self.target

    def fine_grained_binary_search(self, x0, theta, initial_lbd=4.0):
        lbd = initial_lbd
        query = 1
        if not self._is_adversarial(x0 + lbd * theta):
            lbd_lo = lbd
            lbd_hi = lbd * 1.1
            while not self._is_adversarial(x0 + lbd_hi * theta):
                lbd_hi = lbd_hi * 1.1
                query += 1
                if query == 20:
                    return torch.tensor(1e5), query
        else:
            lbd_hi = lbd
            lbd_lo = lbd * 0.9
            while self._is_adversarial(x0 + lbd_lo * theta):
                lbd_lo = lbd_lo * 0.9
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

        # self.th *= 0.996
        return torch.tensor(lbd_hi), query

    def get_loss(self, pert, ldb):
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
            img = image_pert[i]/torch.norm(image_pert[i], p=2)
            _res, query = self.fine_grained_binary_search(self.data, img, ldb)
            distorition = torch.norm(torch.clamp(self.data + _res * img, min=0., max=1.) - self.data, p=2)
            res.append(distorition)
            nquery += query
        return -torch.stack(res), nquery

        # for i in range(len(image_pert)):
        #     _res, query = self.fine_grained_binary_search(self.data, image_pert[i] / torch.norm(image_pert[i], p=2), ldb)
        #     res.append(_res)
        #     nquery += query
        # return -torch.stack(res), nquery

    def __call__(self, pert, ldb):
        return self.get_loss(pert, ldb)

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

class Surrogate_Objective:
    def __init__(self, model, data, target, original_size=32, reduce_size=None, is_targeted=False, nograd=False, mode='bilinear'):
        self.model = model.to(data.device)
        self.original_size = original_size
        self.reduce_size = reduce_size if reduce_size is not None else original_size
        self.is_targeted = is_targeted
        self.data = data
        self.target = target
        self.nograd = nograd
        self.mode = mode

        self.lamda = 1

    def get_logits(self, pert, ldb):
        image_pert = pert.reshape([-1, 3, self.reduce_size, self.reduce_size])
        if self.original_size != self.reduce_size:
            if self.mode == 'bilinear':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='bilinear', align_corners=True)
            elif self.mode == 'nearest':
                image_pert = torch.nn.functional.interpolate(image_pert, [self.original_size, self.original_size], mode='nearest-exact')
            else:
                raise
        image = torch.clamp(self.data + image_pert * ldb, 0, 1)
        res = self.model(image)
        return res.detach() if self.nograd else res

    def get_loss(self, pert, ldb):
        res = cw_loss(self.get_logits(pert, ldb), self.target, is_targeted=self.is_targeted)
        return res.detach() if self.nograd else res

    def __call__(self, pert, ldb):
        return self.get_loss(pert, ldb)

    def get_grad(self, pert, ldb):
        if self.nograd:
            raise NotImplementedError("Cannot get grad if self.nograd is True")
        x = pert.detach()
        x.requires_grad_()
        # obj = self.get_loss(x, eps)
        obj = cw_loss(self.get_logits(pert, ldb), self.target, is_targeted=self.is_targeted)
        obj.backward()
        return x.grad.detach()

    def predict(self, pert, ldb):
        res = self.get_logits(pert, ldb).max(dim=1)[1]
        return res.detach() if self.nograd else res


class PBO_attack(object):
    def __init__(self, args, model, surrogate_model, epsilon, total_images):
        self.model = model
        self.surrogate_model = surrogate_model
        self.dataset_loader = DataLoaderMaker.get_test_attacked_data(args.dataset, args.batch_size)
        self.total_images = total_images    # len(self.dataset_loader.dataset)
        self.batch_size = 1
        self.epsilon = epsilon

        self.query_all = torch.zeros(self.total_images) # query times
        self.distortion_all = defaultdict(OrderedDict)  # key is image index, value is {query: distortion}
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.not_done_all = torch.zeros_like(self.query_all)  # always set to 0 if the original image is misclassified
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.distortion_with_max_queries_all = torch.zeros_like(self.query_all)


    def count_stop_query_and_distortion(self, dist, query):
        if np.sum(dist > self.epsilon) > 0:
            working_ind = np.nonzero(dist > self.epsilon)
            self.success_stop_queries[working_ind] = query[working_ind]
        for inside_batch_index, index_over_all_images in enumerate(self.batch_image_positions):
            self.distortion_all[index_over_all_images][query.item()] = dist

    def run_bo(self, obj, bounds, n_iter, device='cuda', prior_obj=None, scale='adapt'):
        self.success_stop_queries = torch.zeros([self.batch_size]).clone()  # stop query count once the distortion < epsilon
        self.batch_image_positions = np.arange(obj.idx * self.batch_size, min((obj.idx + 1)*self.batch_size, self.total_images)).tolist()
        bo = BayesOpt(obj, bounds, device=device, n_opt=1, n_init=20, n_past=0,
                      normalize_y=True, exact_fval=True, prior_f=prior_obj, scale=scale)
        if scale != 'fixed_only':
            bo.initialize(bo.gen_rand_init() if not obj.is_targeted else obj.target_data)
            if bo.f.get_opt() > 0:
                log.info('Attack succeeded during initialization')
                obj.check_success(bo.f.get_opt_x())
                return bo.f.get_opt_x(), bo.f.call_count
        else:
            bo.initialize(obj.target_data if obj.is_targeted else None)
        self.count_stop_query_and_distortion(abs(bo.f.get_opt()), bo.f.call_count)
        x, _, _ = bo.run(n_iter=n_iter, freq_fit_hyper=10, save_dist=self.count_stop_query_and_distortion)
        log.info('Attack succeeded')
        log.info('Total queries: {}'.format(bo.f.call_count.item()))
        dist = abs(bo.f.get_opt())
        return x, bo.f.call_count, self.success_stop_queries, dist, (dist <= self.epsilon)

    def attack_all_images(self, args, arch_name, result_dump_path):
        reduce_size = args.size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for batch_idx, (images, true_labels) in enumerate(self.dataset_loader):
            set_seed(args.seed)
            if batch_idx < args.start-1:
                continue
            if batch_idx >= args.total_images:
                continue
            if images.size(-1) != self.model.input_size[-1]:
                images = F.interpolate(images, size=self.model.input_size[-1], mode='bilinear', align_corners=False)
            selected = torch.arange(batch_idx * args.batch_size, min((batch_idx + 1) * args.batch_size, self.total_images))
            data, true_label = images.to(device), true_labels.to(device)

            assert data.shape[0] == 1 and true_label.shape[0] == 1
            with torch.no_grad():
                logit = self.model(data)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_label).float()

            if not correct.item():
                log.info(f'Image {batch_idx+1} is classified wrongly')
                continue
            dim = data.shape[0] * data.shape[1] * reduce_size * reduce_size
            bounds = [[-1, 1]] * dim

            if not args.targeted:
                log.info(f'Attack image {batch_idx+1}, original label {true_label.item()}')
                obj = Objective(self.model, batch_idx, data, true_label, original_size=self.model.input_size[-1],
                                reduce_size=reduce_size, is_targeted=args.targeted, mode=args.dr, nograd=True)
                prior_obj = Surrogate_Objective(self.surrogate_model, data, true_label,
                                                original_size=self.model.input_size[-1], reduce_size=reduce_size,
                                                is_targeted=args.targeted, mode=args.dr) if self.surrogate_model else None
                # if args.method == 'bo':
                image_pert, query, success_query, distortion_with_max_queries, success_epsilon = \
                    self.run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)
            else:
                assert true_label.shape[0] == 1
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                  size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    logits = self.model(images)
                    target_labels = logits.argmin(dim=1).to(device)
                elif args.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[args.dataset]).to(device)
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                if args.dataset == "ImageNet":
                    val_dataset = ImageNetDataset(IMAGE_DATA_ROOT[args.dataset], target_labels.item(), "validation")
                elif args.dataset == "CIFAR-10":
                    val_dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[args.dataset], target_labels, "validation")
                val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False)
                target_data, query = list(), 0
                init_num_x = 10
                for i, (xi, yi) in enumerate(val_dataset_loader):
                    if args.dataset == "ImageNet" and self.model.input_size[-1] != 299:
                        xi = F.interpolate(xi, size=(self.model.input_size[-2], self.model.input_size[-1]),
                                           mode='bilinear', align_corners=False)
                    xi = xi.to(device)
                    yi_pred = self.model(xi).max(1)[1]
                    query += 1
                    if yi_pred != target_labels:
                        continue
                    theta = xi - data
                    theta = F.interpolate(theta, size=(reduce_size, reduce_size), mode='bilinear', align_corners=False)
                    target_data.append(theta.view(1, -1)/torch.norm(theta.view(-1), p=2))
                    if len(target_data) == init_num_x:
                        break
                target_data = torch.cat(target_data, dim=0)
                # print(target_data.shape)

                log.info(f'Attack image {batch_idx+1}, original label {true_label.item()}, new label {target_labels}')
                obj = Objective(self.model, data, target_labels, target_data, original_size=self.model.input_size[-1], reduce_size=reduce_size,
                                is_targeted=args.targeted, mode=args.dr, nograd=True, init_q=query)
                prior_obj = Objective(self.surrogate_model, data, target_labels, reduce_size=reduce_size,
                                      is_targeted=args.targeted, mode=args.dr) if self.surrogate_model else None
                # if args.method == 'bo':
                image_pert, query, success_query, distortion_with_max_queries, success_epsilon = \
                    self.run_bo(obj, bounds, args.max_queries, device=device, prior_obj=prior_obj, scale=args.bo_scale)
            query = query.cuda()
            success_query = torch.tensor(success_query).float()
            distortion_with_max_queries = torch.tensor(distortion_with_max_queries).float()
            success_epsilon = torch.tensor(success_epsilon).float()
            with torch.no_grad():
                adv_logit = obj.get_logits(image_pert)
            adv_pred = adv_logit.argmax(dim=1)
            ## Continue query count
            not_done = correct.clone()
            if args.targeted:
                not_done = not_done * (1 - adv_pred.eq(target_labels.cuda()).float()).float()  # not_done初始化为 correct, shape = (batch_size,)
            else:
                not_done = not_done * adv_pred.eq(true_labels.cuda()).float()  #
            success = (1 - not_done.detach().cpu()) * correct.detach().cpu() * success_epsilon.detach().cpu() * (
                    success_query.detach().cpu() <= args.max_queries).float()

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
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def set_seed(seed):
    torch.backends.cudnn.enabled = True
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
    parser.add_argument('--max-queries', type=int, default=10000)
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--json-config', type=str, default='TangentAttack-main/configures/PBO.json',
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
    parser.add_argument('--load-random-class-image', action='store_true', help='load a random image from the target class')

    parser.add_argument('--method', default='bo', type=str, choices=['bo'],
                        help='to use bayesian optimization (bo)')
    parser.add_argument('--bo-scale', default='adapt', type=str, choices=['fixed', 'adapt', 'fixed_only'],
                        help='how to use surrogate models to guide optimization, see the doc in bo.py')
    parser.add_argument('--size', default=56, type=int, help='default=32: not using dimension reduction.'
                        'If <32, then reduce dimension to size*size*3')
    parser.add_argument('--start', type=int, default=0, help='skipping the first `start` images')
    parser.add_argument('--dr', type=str, default='bilinear', choices=['bilinear', 'nearest'], help='the dimension reduction algorithm')
    parser.add_argument('--total_images', type=int, default=100)


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
        defaults = json.load(open(os.path.join(PROJECT_PATH, args.json_config)))[args.dataset][args.norm]
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

    # if args.all_archs:
    #     archs = []
    #     if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
    #         for arch in MODELS_TEST_STANDARD[args.dataset]:
    #             test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(PY_ROOT,
    #                                                                                     args.dataset,  arch)
    #             if os.path.exists(test_model_path):
    #                 archs.append(arch)
    #             else:
    #                 log.info(test_model_path + " does not exists!")
    #     elif args.dataset == "TinyImageNet":
    #         for arch in MODELS_TEST_STANDARD[args.dataset]:
    #             test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
    #                 root=PY_ROOT, dataset=args.dataset, arch=arch)
    #             test_model_path = list(glob.glob(test_model_list_path))
    #             if test_model_path and os.path.exists(test_model_path[0]):
    #                 archs.append(arch)
    #     else:
    #         for arch in MODELS_TEST_STANDARD[args.dataset]:
    #             test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
    #                 PY_ROOT,
    #                 args.dataset, arch)
    #             test_model_list_path = list(glob.glob(test_model_list_path))
    #             if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
    #                 continue
    #             archs.append(arch)
    # else:
    #     assert args.arch is not None
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
        surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False) if args.surrogate_arch is not None else None
        attacker = PBO_attack(args, model, surrogate_model, args.epsilon, total_images=args.total_images)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()
