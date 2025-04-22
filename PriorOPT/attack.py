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
from config import MODELS_TEST_STANDARD
from PriorOPT.prior_opt_l2_norm_attack import PriorOptL2Norm
from PriorOPT.prior_opt_l2_norm_targeted_attack import PriorOptL2NormTargetedAttack
from PriorOPT.prior_opt_linf_norm_attack import PriorOptLinfNorm
from models.defensive_model import DefensiveModel
from models.standard_model import StandardModel


def get_exp_dir_name(dataset,  norm, targeted, target_type, args):
    if target_type == "load_random":
        target_type = "random"
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    if args.ablation_study:
        if args.sign:
            dirname = 'PriorSignOPT-{}-{}-{}/ablation_study'.format(dataset, norm, target_str)
        else:
            dirname = 'PriorOPT-{}-{}-{}/ablation_study'.format(dataset, norm, target_str)
        return dirname

    if args.best_initial_target_sample:
        if args.sign:
            if args.attack_defense:
                dirname = 'PriorSignOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'PriorSignOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        else:
            if args.attack_defense:
                dirname = 'PriorOPT_best_start_initial_on_defensive_model-{}-{}-{}'.format(dataset, norm, target_str)
            else:
                dirname = 'PriorOPT_best_start_initial-{}-{}-{}'.format(dataset, norm, target_str)
        return dirname
    if args.sign:
        if args.attack_defense:
            dirname = 'PriorSignOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'PriorSignOPT-{}-{}-{}'.format(dataset, norm, target_str)
    else:
        if args.attack_defense:
            dirname = 'PriorOPT_on_defensive_model-{}-{}-{}'.format(dataset,  norm, target_str)
        else:
            dirname = 'PriorOPT-{}-{}-{}'.format(dataset, norm, target_str)
    if args.PGD_init_theta:
        dirname += '_with_PGD_init_theta'
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
    parser.add_argument('--json-config', type=str, default='./configures/PriorOPT.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--norm', type=str, required=True, choices=["l2","linf"], help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--est-grad-samples', type=int,default=200)
    parser.add_argument('--epsilon', type=float,
                        help='epsilon of the maximum perturbation in l_p norm attack')
    parser.add_argument('--batch-size', type=int, default=1,  help='test batch size')
    parser.add_argument('--exp-dir', default='logs', type=str, help='directory to save results and logs')
    parser.add_argument('--arch', default=None, type=str, help='network architecture')
    surrogate_arch_group = parser.add_mutually_exclusive_group(required=True)
    surrogate_arch_group.add_argument("--surrogate-arch", type=str, help="the architecture of a surrogate model")
    surrogate_arch_group.add_argument("--surrogate-archs", nargs="+", help="multiple surrogate models, and this parameter should be passed in through space splitting")
    parser.add_argument('--all_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random','load_random', 'least_likely', "increment"])
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--gpu', type=int, required=True, help='which GPU ID will be used')
    parser.add_argument('--attack-defense', action="store_true")
    parser.add_argument('--defense-model', type=str, default=None)
    parser.add_argument('--defense-norm', type=str, choices=["l2", "linf"], default='linf')
    parser.add_argument('--defense-eps', type=str,default="")
    parser.add_argument('--best-initial-target-sample', action='store_true',
                        help='Using a target image with the shortest distortion as the initial images. '
                             'By default (do not pass --best_initial_target_sample), '
                             'we use a random selected target image as the initial sample')
    parser.add_argument('--sign',action='store_true',help="whether to use the sign-based prior-guided gradient estimation")
    parser.add_argument('--clip-grad-max-norm',type=float)
    parser.add_argument('--tol',type=float,)
    parser.add_argument('--ablation-study',action='store_true')
    parser.add_argument('--prior-grad-bs-tol', type=float,default=0.01, help="the binary search's stopping threshold for estimating gradient")
    parser.add_argument('--PGD-init-theta', action="store_true")

    # parser.add_argument('--alpha', type=float)
    args = parser.parse_args()
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
    return args

if __name__ == "__main__":
    args = get_parse_args()
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        targeted_attack_str = "targeted" if args.targeted else "untargeted"
        defaults = json.load(open(args.json_config))[args.dataset][args.norm][targeted_attack_str]
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
            if args.dataset == "ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_defense_{}_{}_{}_{}.log".format(args.arch, args.defense_model,
                                                                               args.defense_norm, args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    if args.surrogate_archs is not None:
        if args.attack_defense:
            if args.dataset == "ImageNet":
                log_file_path = osp.join(args.exp_dir,
                                         "run_{}_surrogates_{}_defense_{}_{}_{}.log".format(args.arch, ",".join(args.surrogate_archs),
                                                                 args.defense_model, args.defense_norm, args.defense_eps))
            else:
                log_file_path = osp.join(args.exp_dir, 'run_{}_surrogates_{}_defense_{}.log'.format(args.arch,  ",".join(args.surrogate_archs),
                                                                                                    args.defense_model))
        else:
            log_file_path = osp.join(args.exp_dir, 'run_{}_surrogates_{}.log'.format(args.arch,  ",".join(args.surrogate_archs)))
    if args.ablation_study:
        log_file_path = osp.join(args.exp_dir, 'run_{}_estimate_grad_samples_{}.log'.format(args.arch, args.est_grad_samples))
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
    surrogate_models = []
    if args.surrogate_arch is not None:
        surrogate_model = StandardModel(args.dataset, args.surrogate_arch, False, load_pretrained=True)
        surrogate_model.cuda()
        surrogate_model.eval()
        surrogate_models.append(surrogate_model)
    else:
        for surrogate_arch in args.surrogate_archs:
            surrogate_model = StandardModel(args.dataset, surrogate_arch, False,
                                            load_pretrained=True)
            surrogate_model.cuda()
            surrogate_model.eval()
            surrogate_models.append(surrogate_model)
    for arch in archs:
        if args.attack_defense:
            if args.defense_model == "adv_train_on_ImageNet":
                save_result_path = args.exp_dir + "/{}_{}_{}_{}_result.json".format(arch, args.defense_model,
                                                                                    args.defense_norm, args.defense_eps)
            else:
                save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if args.surrogate_archs is not None:
            if args.attack_defense:
                if args.defense_model == "adv_train_on_ImageNet":
                    save_result_path = args.exp_dir + "/{}_surrogates_{}_{}_{}_{}_result.json".format(arch,  ",".join(args.surrogate_archs),
                                                                             args.defense_model, args.defense_norm, args.defense_eps)
                else:
                    save_result_path = args.exp_dir + "/{}_surrogates_{}_{}_result.json".format(arch, ",".join(args.surrogate_archs),
                                                                                                args.defense_model)
            else:
                save_result_path = args.exp_dir + "/{}_surrogates_{}_result.json".format(arch, ",".join(args.surrogate_archs))

        if args.ablation_study:
            save_result_path = args.exp_dir + "/{}_with_{}_grad_samples_result.json".format(arch, args.est_grad_samples)
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model,norm=args.defense_norm, eps=args.defense_eps)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        for surrogate_model in surrogate_models:
            assert model.input_size[-1] == surrogate_model.input_size[-1], "Error, the input sizes (image's dimension) of the target model and the surrogate model do not equal!"
        tol = None
        if args.tol is not None and args.tol != 0.0:
            tol = args.tol
        if args.norm == "l2":
            if args.targeted:
                attacker = PriorOptL2Norm(model, surrogate_models, args.dataset, args.epsilon, args.targeted,
                                     args.batch_size, args.est_grad_samples,maximum_queries=args.max_queries, sign=args.sign, clip_grad_max_norm=args.clip_grad_max_norm,
                                      tol=tol, prior_grad_binary_search_tol=args.prior_grad_bs_tol, best_initial_target_sample=args.best_initial_target_sample,
                                      PGD_init_theta=args.PGD_init_theta)
            else:
                attacker = PriorOptL2Norm(model, surrogate_models, args.dataset, args.epsilon, args.targeted,
                                      args.batch_size, args.est_grad_samples,
                                      maximum_queries=args.max_queries, sign=args.sign, clip_grad_max_norm=args.clip_grad_max_norm,
                                      tol=tol, prior_grad_binary_search_tol=args.prior_grad_bs_tol, best_initial_target_sample=args.best_initial_target_sample,
                                      PGD_init_theta=args.PGD_init_theta)
            attacker.attack_all_images(args, arch, save_result_path)
        elif args.norm == "linf":
            attacker = PriorOptLinfNorm(model, surrogate_models, args.dataset, args.epsilon, args.targeted,
                                        args.batch_size, args.est_grad_samples, maximum_queries=args.max_queries,
                                         sign=args.sign, clip_grad_max_norm=args.clip_grad_max_norm, tol=tol,
                                        prior_grad_binary_search_tol=args.prior_grad_bs_tol, best_initial_target_sample=args.best_initial_target_sample)
            attacker.attack_all_images(args, arch, save_result_path)
        model.cpu()

