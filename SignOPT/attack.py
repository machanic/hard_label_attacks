import random
import sys
import os


sys.path.append(os.getcwd())
import json
from types import SimpleNamespace
import torch
import argparse
import numpy as np
import os.path as osp
import glog as log
from config import MODELS_TEST_STANDARD, IN_CHANNELS
from dataset.dataset_loader_maker import DataLoaderMaker
from SignOPT.sign_opt_l2_norm_attack import SignOptL2Norm
from SignOPT.sign_opt_linf_norm_attack import SignOptLinfNorm
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel


def distance(x_adv, x, norm='l2'):
    diff = (x_adv - x).view(x.size(0), -1)
    if norm == 'l2':
        out = torch.sqrt(torch.sum(diff * diff)).item()
        return out
    elif norm == 'linf':
        out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out

def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.best_initial_target_sample:
        if args.svm:
            if args.attack_defense:
                dirname = 'SVMOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'SVMOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            if args.attack_defense:
                dirname = 'SignOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'SignOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        return dirname
    if args.svm:
        if args.attack_defense:
            dirname = 'SVMOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'SVMOPT-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'SignOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'SignOPT-{}-{}-{}'.format(dataset, norm, target_str)
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

def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["CIFAR-10","CIFAR-100","ImageNet","TinyImageNet"],
                        help='Dataset to be used, [CIFAR-10, CIFAR-100, ImageNet, TinyImageNet]')
    parser.add_argument('--json-config', type=str, default='./configures/SignOPT.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--norm', type=str, required=True, choices=["l2","linf"], help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--est-grad-queries', type=int,default=200)
    parser.add_argument('--epsilon', type=float,
                        help='epsilon of the maximum perturbation in l_p norm attack')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='verbose.')
    parser.add_argument('--batch_size', type=int, default=1,  help='test batch size')
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random','load_random', 'least_likely', "increment"])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--gpu', type=int, required=True, help='which GPU ID will be used')
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--defense_norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense_eps', type=str,default="")
    parser.add_argument('--best-initial-target-sample', action='store_true',
                        help='Using a target image with the shortest distortion as the initial images. '
                             'By default (do not pass --best_initial_target_sample), '
                             'we use a random selected target image as the initial sample')
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')
    parser.add_argument('--svm',action='store_true',help="using this option is SVM-OPT attack")
    parser.add_argument('--tol',type=float,)
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args

if __name__ == "__main__":
    args = get_parse_args()
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm]
        arg_vars = vars(args)
        arg_vars_ = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars_)
        defaults.update({k: v for k, v in arg_vars.items() if k not in defaults})
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted and args.dataset == "ImageNet":
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
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        tol = None
        if args.tol is not None and args.tol !=0.0:
            tol = args.tol
        if args.norm == "l2":
            attacker = SignOptL2Norm(model, args.dataset, args.epsilon, args.targeted,
                                     args.batch_size, args.est_grad_queries,
                                    maximum_queries=args.max_queries,svm=args.svm,tol=tol,
                                     best_initial_target_sample=args.best_initial_target_sample)
            attacker.attack_all_images(args, arch, save_result_path)
        elif args.norm == "linf":
            attacker = SignOptLinfNorm(model, args.dataset, args.epsilon, args.targeted, args.batch_size,
                                       args.est_grad_queries, maximum_queries=args.max_queries, svm=args.svm, tol=tol,
                                       best_initial_target_sample=args.best_initial_target_sample)
            attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()

