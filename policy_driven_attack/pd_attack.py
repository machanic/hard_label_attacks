#!/usr/bin/env python3
import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
import string
import functools
import socket
import getpass
import argparse
import collections
import json
import random
import math
import itertools
from collections import OrderedDict
import copy
from types import SimpleNamespace
from config import CLASS_NUM, MODELS_TEST_STANDARD, IN_CHANNELS, IMAGE_DATA_ROOT, PROJECT_PATH
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from pda_models import make_policy_model
from models.standard_model import StandardModel
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.target_class_dataset import ImageNetDataset, CIFAR100Dataset, CIFAR10Dataset, TinyImageNetDataset
from policy_driven_attack.pda_models.victim import Standard, VictimQuery

class PDAttack(object):
    def __init__(self, victim_query, epsilon, external_init_adv_image, use_pytorch_rng, ssh, dataset, attack_type,
                 victim_arch, phase='test', norm_type='l2', victim_batch_size=50,
                 clip_min=0, clip_max=1.0, batch_size=1):
        self.victim_query = victim_query
        self.epsilon = epsilon
        self.dataset = dataset
        self.phase = phase
        self.ssh = ssh
        self.norm_type = norm_type
        self.victim_batch_size = victim_batch_size
        self.external_init_adv_image = external_init_adv_image
        self.victim_arch = victim_arch
        self.attack_type = attack_type
        self.use_pytorch_rng = use_pytorch_rng
        self.num_of_step = 0
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.images_index = 0
        if args.save_grad:
            loader = DataLoaderMaker.get_imgid_img_label_data_loader(dataset, batch_size, True, True)
        else:
            loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, victim_query.net.arch)
        self.dataset_loader = loader
        self.total_images = len(self.dataset_loader.dataset)
        self.load_random_class_image = args.load_random_class_image

    def calc_distance(self, x1, x2):
        diff = x1.cuda() - x2.cuda()
        diff = diff.view(diff.shape[0], -1)
        if self.norm_type == 'l2':
            return torch.sqrt((diff ** 2).sum(dim=1))
        elif self.norm_type == 'linf':
            return diff.abs().max(dim=1)[0]
        else:
            raise NotImplementedError('Unknown norm: {}'.format(self.norm_type))

    def get_image_of_target_class(self, dataset_name, target_labels):

        images = []
        for label in target_labels:  # length of target_labels is 1
            if self.load_random_class_image:
                initial_images = np.load(
                    "{}/attacked_images/{}/{}_targeted-attack-initial-images.npz".format(PROJECT_PATH, dataset_name,
                                                                                         dataset_name),
                    allow_pickle=True)
                image = torch.from_numpy(initial_images[str(label.item())]).unsqueeze(0)
                if dataset_name == "ImageNet" and self.victim_query.net.input_size[-1] != 299:
                    image = F.interpolate(image,
                                          size=(self.victim_query.net.input_size[-2], self.victim_query.net.input_size[-1]),
                                          mode='bicubic',
                                          align_corners=False)

                with torch.no_grad():
                    testlogits = self.victim_query.query(image.cuda(), True, True)# --[debug]
                    print(testlogits)

            else:
                if dataset_name == "ImageNet":
                    dataset = ImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
                elif dataset_name == "CIFAR-10":
                    dataset = CIFAR10Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
                elif dataset_name == "CIFAR-100":
                    dataset = CIFAR100Dataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
                elif dataset_name == "TinyImageNet":
                    dataset = TinyImageNetDataset(IMAGE_DATA_ROOT[dataset_name], label.item(), "validation")
                index = np.random.randint(0, len(dataset))
                image, true_label = dataset[index]
                image = image.unsqueeze(0)
                if dataset_name == "ImageNet" and self.victim_query.net.input_size[-1] != 299:
                    image = F.interpolate(image,
                                          size=(self.victim_query.net.input_size[-2], self.victim_query.net.input_size[-1]),
                                          mode='bicubic', align_corners=False)
                with torch.no_grad():
                    logits = self.victim_query.query(image.cuda(), True, True)# --[debug]

                max_recursive_loop_limit = 100
                loop_count = 0
                while not logits.item() and loop_count < max_recursive_loop_limit:
                    loop_count += 1
                    index = np.random.randint(0, len(dataset))
                    image, true_label = dataset[index]
                    image = image.unsqueeze(0)
                    if dataset_name == "ImageNet" and self.victim_query.net.input_size[-1] != 299:
                        image = F.interpolate(image,
                                              size=(self.victim_query.net.input_size[-2], self.victim_query.net.input_size[-1]),
                                              mode='bicubic', align_corners=False)
                    with torch.no_grad():
                        logits = self.victim_query.query(image.cuda(), True, True)

                if loop_count == max_recursive_loop_limit:
                    # The program cannot find a valid image from the validation set.
                    return None
                assert true_label == label.item()
            images.append(torch.squeeze(image))

        return torch.stack(images)  # B,C,H,W

    def project(self, unperturbed, perturbed_inputs, alphas):
        alphas = alphas.view(-1, 1, 1, 1)
        if self.norm_type == 'l2':
            projected = (1 - alphas.cuda()) * unperturbed.cuda() + alphas.cuda() * perturbed_inputs.cuda()
        elif self.norm_type == 'linf':
            projected = torch.max(perturbed_inputs, unperturbed - alphas)
            projected = torch.min(projected, unperturbed + alphas)
        else:
            raise NotImplementedError('Unknown norm: {}'.format(self.norm_type))
        return projected

    def success(self, images):
        images = torch.clamp(images, min=self.clip_min, max=self.clip_max).cuda()
        if len(images.shape) == 5:
            images = images[0]
        logits = self.victim_query.query(images, sync_best=True, no_count=False)
        return logits.detach().cuda().item()

    def decision_function(self, victim, x, sync_best=True, no_count=False):
        outs = list()
        num_batchs = int(math.ceil(float(x.shape[0]) / self.victim_batch_size))
        for j in range(num_batchs):
            current_batch = x[self.victim_batch_size * j:self.victim_batch_size * (j + 1)]
            out = victim.query(current_batch, sync_best=sync_best, no_count=no_count)
            outs.append(out)
        return torch.cat(outs)

    # initialization for the attack
    def initialize(self, sample, target_images, true_labels, target_labels):
        """
        sample: the shape of sample is [C,H,W] without batch-size
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        num_eval = 0
        if target_images is None:
            while True:
                random_noise = torch.from_numpy(np.random.uniform(self.clip_min, self.clip_max,
                                                                  size=(IN_CHANNELS[args.dataset], self.victim_query.net.input_size[-2], self.victim_query.net.input_size[-1]))).float()
                # random_noise = torch.FloatTensor(*self.shape).uniform_(self.clip_min, self.clip_max)
                success = self.success(random_noise[None])
                num_eval += 1
                if success:
                    break
                if num_eval > 1000:
                    log.info("Initialization failed! Use a misclassified image as `target_image")
                    if target_labels is None:
                        target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                      size=true_labels.size()).long()
                        invalid_target_index = target_labels.cuda().eq(true_labels.cuda())
                        while invalid_target_index.sum().item() > 0:
                            target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                                size=target_labels[invalid_target_index].cuda().size()).long()
                            invalid_target_index = target_labels.cuda().eq(true_labels.cuda())

                    initialization = self.get_image_of_target_class(self.dataset,target_labels).squeeze()
                    return initialization, 1
                # assert num_eval < 1e4, "Initialization failed! Use a misclassified image as `target_image`"
            # Binary search to minimize l2 distance to original image.
            low = 0.0
            high = 1.0
            while high - low > 0.001:
                mid = (high + low) / 2.0
                blended = (1 - mid) * sample + mid * random_noise.cuda()
                success = self.success(blended[None])
                num_eval += 1
                if success:
                    high = mid
                else:
                    low = mid
            # Sometimes, the found `high` is so tiny that the difference between initialization and sample is very very small, this case will cause inifinity loop
            initialization = (1 - high) * sample + high * random_noise.cuda()

        else:
            initialization = target_images

        return initialization, num_eval

    def binary_search(self, victim, image, adv_image, theta, no_count=False):
        # Compute distance between each of perturbed and unperturbed input.
        distance = self.calc_distance(image, adv_image)

        # Choose upper threshold in binary search based on constraint.
        if self.norm_type == 'linf':
            high = distance
            # Stopping criteria.
            threshold = torch.clamp(distance * theta, max=theta)
        else:
            assert self.norm_type == 'l2'
            high = torch.ones(adv_image.shape[0]).to(adv_image.device)
            threshold = theta

        # Call recursive function.
        low = torch.zeros_like(high)
        high = high.to(torch.float64)
        low = low.to(torch.float64)
        old_mid = high.cuda()
        while ((high - low) / threshold).max().item() > 1:
            # projection to mids.
            mid = (high + low) / 2.0
            mid_image = self.project(image, adv_image, mid)

            # Update high and low based on model decisions.
            decision = self.decision_function(victim, mid_image, no_count=no_count)
            high = torch.where(decision.cuda(), mid.cuda(), high.cuda())
            low = torch.where(~decision.cuda(), mid.cuda(), low.cuda())
            reached_numerical_precision = (old_mid == mid).all()
            old_mid = mid
            if reached_numerical_precision:
                break

        return self.project(image, adv_image, high)

    def norm(self, v, p=2):
        v = v.view(v.shape[0], -1)
        if self.dataset == 'debug':
            output_shape = (-1, 1)
        else:
            output_shape = (-1, 1, 1, 1)
        if p == 2:
            return torch.clamp(v.norm(dim=1).view(*output_shape), min=1e-8)
        elif p == 1:
            return torch.clamp(v.abs().sum(dim=1).view(*output_shape), min=1e-8)
        else:
            raise ValueError('Unknown norm p={}'.format(p))

    def bubble_sort(self, array):
        for i in range(1, len(array)):
            for j in range(0, len(array) - i):
                if int(array[j][0]) > int(array[j + 1][0]):
                    array[j], array[j + 1] = array[j + 1], array[j]
        return array

    def cw_loss(self, logit, label, target):
        _, argsort = logit.sort(dim=1, descending=True)
        label_logit = logit[torch.arange(logit.shape[0]).long(), label.long()]
        target_logit = logit[torch.arange(logit.shape[0]).long(), target.long()]
        assert not label.eq(target).any().item()
        assert target_logit.ge(label_logit).all().item()
        return target_logit - label_logit

    def attack_all_images(self, args):
        # make model
        victim = self.victim_query

        # make loader
        kwargs = dict()
        if args.dataset == 'ImageNet':
            kwargs['size'] = victim.input_size[-1]
        # make policy model
        kwargs = OrderedDict({'init_std': args.policy_init_std, 'input_size': args.grad_size})
        if len(args.policy_weight_fname) > 0:
            if args.policy_arch == 'empty':
                log.info(
                    'Ignore args.policy_weight_fname: {}, since policy arch is empty'.format(args.policy_weight_fname))
            else:
                kwargs['weight_fname'] = args.policy_weight_fname

        if args.policy_arch in ['unet']:
            kwargs['bilinear'] = args.policy_bilinear
            kwargs['normalization_type'] = args.policy_normalization_type
            kwargs['base_width'] = args.policy_base_width
        elif args.policy_arch.endswith('_inv'):
            kwargs['calibrate'] = args.policy_calibrate
            kwargs['use_tanh'] = args.policy_use_tanh
            if args.policy_arch.startswith('vgg') or args.policy_arch.startswith('resnet'):
                kwargs['normalization_type'] = args.policy_normalization_type

        policy_keys = OrderedDict()
        for key, value in vars(args).items():
            if key.startswith('policy_') and key != 'policy_arch':
                policy_keys[key] = value
        log.info('Found policy keys: {}, only use {} for dataset {} and arch {}'.format(
            policy_keys, kwargs, args.dataset, args.policy_arch))
        policy = make_policy_model(args.dataset, args.policy_arch, **kwargs).train().to(device)
        log.info('Policy network:')
        log.info(policy)

        # output these fields from policy network
        if args.policy_arch.endswith('_inv'):
            output_fields = ('grad', 'std', 'adv_logit', 'logit')
        else:
            output_fields = ('grad', 'std')

        # make upsampler and downsampler

        if args.grad_size != 0:
            # upsampler: grad to image; downsampler: image to grad
            upsampler = lambda x: F.interpolate(x, size=victim.input_size[-1], mode='bicubic' if args.dataset=="ImageNet" else "bilinear", align_corners=True)
            downsampler = lambda x: F.interpolate(x, size=args.grad_size, mode='bicubic' if args.dataset=="ImageNet" else "bilinear", align_corners=True)
        else:
            # no resize, upsampler = downsampler = identity
            upsampler = downsampler = lambda x: x
        # load previously used image ids when training gradient model, if there are any
        used_image_ids = OrderedDict()
        used_image_ids['train_seen'] = list()
        used_image_ids['train_unseen'] = list()
        used_image_ids['test'] = list()
        if len(args.image_id_ref) > 0:
            with open(osp.join(args.image_id_ref, 'config.json'), 'r') as f:
                image_id_ref_config = json.load(f)
            if image_id_ref_config['dataset'] != args.dataset or image_id_ref_config['phase'] != args.phase:
                log.info('Ignore grad model image ids from {}, because they use dataset {} and phase {}'.format(
                    args.image_id_ref, image_id_ref_config['dataset'], image_id_ref_config['phase']))
            else:
                for key in used_image_ids.keys():
                    fname = osp.join(args.image_id_ref, 'results', '{}_image_ids.pth'.format(key))
                    used_image_ids[key] = torch.load(fname).tolist()
                log.info('Load used image ids from {}'.format(args.image_id_ref))
        for key, image_ids in used_image_ids.items():
            log.info('Found {} used image ids, key: {}'.format(len(image_ids), key))

        # make optimizer
        def trainable(name):
            if name.split('.')[-1] in ['normal_logstd']:
                return False
            else:
                return True

        param_groups = list()
        param_groups.append(
            {'params': [p[1] for p in policy.named_parameters() if trainable(p[0]) and 'bias' not in p[0]],
             'lr': args.lr, 'weight_decay': args.decay})
        param_groups.append({'params': [p[1] for p in policy.named_parameters() if trainable(p[0]) and 'bias' in p[0]],
                             'lr': args.lr, 'weight_decay': 0.0})
        if not args.exclude_std:
            param_groups.append({'params': [p[1] for p in policy.named_parameters()
                                            if 'normal_logstd' in p[0] and trainable(p[0])],
                                 'lr': args.lr * args.std_lr_mult, 'weight_decay': 0.0})
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(param_groups)
        else:
            raise ValueError('Unknown optimizer: {}'.format(args.optimizer))

        optimizer_init_state_dict = copy.deepcopy(optimizer.state_dict())
        log.info('Optimizer: {}'.format(optimizer))
        log.info('Optimizer init state: {}'.format(optimizer_init_state_dict))
        log.info('Number of params to be optimized: {}'.format(
            [len(param_group['params']) for param_group in param_groups]))

        # start attack

        # these four variables represent type of visited images, and we treat them as boolean tensors
        # we use LongTensor instead of ByteTensor because we will do is_x.sum() later and ByteTensor will overflow
        is_correct = torch.LongTensor(0)
        is_ignore = torch.LongTensor(0)
        is_image_type = OrderedDict([
            ('train_seen', torch.LongTensor(0)),
            ('train_unseen', torch.LongTensor(0)),
            ('val', torch.LongTensor(0)),
            ('test', torch.LongTensor(0))
        ])
        num_image_each_type = OrderedDict([
            ('train_seen', min(args.num_image, len(used_image_ids['train_seen']))),
            ('train_unseen', 0),
            ('val', 0),
            ('test', args.num_image)
        ])
        log.info('Number of images included in the attacking task:')
        for key in num_image_each_type.keys():
            log.info('    {}: {}'.format(key, num_image_each_type[key]))

        # print function we call in the end of each iteration and when whole attack ends
        def print_progress(title):
            # print attack progress
            log.info(title)
            log.info('  visited images: {}'.format(is_correct.numel()))
            log.info('  ignored images: {}'.format(is_ignore.sum().item()))
            log.info('  correct images: {}'.format(is_correct.sum().item()))
            for key in is_image_type.keys():
                log.info('  correct {} images: {} not ignored, {} in total'.format(
                    key, (is_correct & is_image_type[key] & (1 - is_ignore)).sum().item(),
                    (is_correct & is_image_type[key]).sum().item()))

        # perform pre tune
        def do_pre_tune(adv_image_, image_, label_, target_):
            if args.num_pre_tune_step == 0:
                log.info('No pre-tune applied since args.num_pre_tune_step is 0')
                return
            if args.grad_method != 'policy_distance':
                log.info('No pre-tune applied since args.grad_method is not policy_distance')
                return
            if not args.policy_arch.endswith('_inv'):
                log.info('No pre-tune applied since args.policy_arch is not an inverse network')
                return

            # make optimizer
            optimizer_ce_tce_lce = torch.optim.SGD([p[1] for p in policy.named_parameters()
                                                    if 'normal_logstd' not in p[0] and trainable(p[0])],
                                                   lr=args.pre_tune_lr, momentum=0.9, weight_decay=0.001)

            # clear existing grads
            policy.zero_grad()
            optimizer_ce_tce_lce.zero_grad()

            # start optimize args.num_pre_tune_step steps
            log.info('Now do pre-tune for {} steps, or until all ce values <= {:.4f}'.format(
                args.num_pre_tune_step, args.pre_tune_th))
            output_fields_ = ('grad', 'adv_logit', 'logit')
            for step_index_ in range(args.num_pre_tune_step):
                output_ = policy(adv_image_, image_, label_, target_, output_fields=output_fields_)
                # logit for adv_image
                adv_logit_ = output_['adv_logit']
                lce_ = F.cross_entropy(adv_logit_, label_.view(-1), reduction='none')
                tce_ = F.cross_entropy(adv_logit_, target_.view(-1), reduction='none')

                # logit for clean_image
                logit_ = output_['logit']
                ce_ = F.cross_entropy(logit_, label_.view(-1), reduction='none')
                log.info('Pre-tune step {}: ce {:.4f}, lce {:.4f}, tce {:.4f}'.format(
                    step_index_ + 1, ce_.item(), lce_.item(), tce_.item()))

                # early break if ce, tce and lce are all small enough
                if max([tce_.item(), lce_.item(), ce_.item()]) <= args.pre_tune_th:
                    break

                # make loss, bp, update, and clear gradient
                loss_ = ce_ + 0.5 * tce_ + 0.5 * lce_
                loss_.backward()
                optimizer_ce_tce_lce.step()
                policy.zero_grad()
                optimizer_ce_tce_lce.zero_grad()
            log.info('Pre-tune finished')

        ######################################
        alld = {}
        correct_all = []
        success_all = []
        notdone_all = []
        query_all = []
        success_query_all = []
        distortion_with_max_querys = []
        num_of_image_circle = 0
        num_of_index = 0
        data = {}
        data_for_labels = {}
        if args.save_grad:
            with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                               'image_ids_step.json'), 'w') as f:
                json.dump(data, f)
            with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                               'images_and_labels.json'), 'w') as f:
                json.dump(data_for_labels, f)
            with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                               'image_ids_index.json'), 'w') as f:
                json.dump(data_for_labels, f)
        ######################################
        size_of_image = victim.input_size[-1]
        channel = victim.input_size[0]
        # start to attack images, iterate over data loader
        for batch_index, (image_id, image, label) in enumerate(self.dataset_loader):
            # batch attack is not supported yet
            assert image_id.numel() == 1

            # make sure the shape of image is correct for victim
            if tuple(image.shape[1:]) != tuple(victim.input_size) and args.dataset == 'ImageNet':
                image = F.interpolate(image, size=(victim.input_size[-2], victim.input_size[-1]), mode='bicubic',
                                      align_corners=False)
            assert tuple(image.shape[1:]) == tuple(victim.input_size)

            distortion = {}
            query_list = []
            distortion_list = []

            # append 0, and we will modify them later
            is_correct = torch.cat((is_correct, torch.LongTensor([0])))
            is_ignore = torch.cat((is_ignore, torch.LongTensor([0])))
            for image_type in is_image_type.keys():
                is_image_type[image_type] = torch.cat((is_image_type[image_type], torch.LongTensor([0])))

            # determine type of this image, use used_image_ids to determine the type
            if image_id.item() in used_image_ids['train_seen']:
                # this image was used as training set in train_grad_model.py
                image_type = 'train_seen'
            elif image_id.item() in used_image_ids['train_unseen']:
                # this image was not used as training set in train_grad_model.py
                # sometimes we also use these images to select the best model, so we can also treat them as 'val'
                # image_type = 'train_unseen'
                image_type = 'val'
            elif image_id.item() in used_image_ids['test']:
                # this image was used to select the best model in train_grad_model.py
                image_type = 'val'
            else:
                # this image is brand new
                image_type = 'test'
            is_image_type[image_type][-1] = 1

            # freeze bn to use previously estimated mean/var instead of current batch mean/var
            for m in policy.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    m.eval()

            # given the shape of input image, set the value of binary search threshold
            d = image.numel() / image.shape[0]

            if args.norm_type == 'l2':
                # theta = args.gamma / math.sqrt(d)
                theta = args.gamma / math.sqrt(d) / d
            elif args.norm_type == 'linf':
                # theta = args.gamma / d
                theta = args.gamma / d ** 2
            else:
                raise NotImplementedError('Unknown norm type: {}'.format(args.norm_type))

            # move inputs to device
            image = image.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred = victim(image).argmax(dim=1)
            acc = pred.eq(label)
            is_correct[-1] = acc.item()

            # check whether we have visit enough images
            # we do not check this in the beginning of loop since values of is_* variables are not determined
            # if we have visited num_train_seen_image train images then we should ignore train_seen images later
            # if we have visited num_train_unseen_image train images then we should ignore train_unseen images later
            # if we have visited num_val_image val images then we should ignore val images later
            # if we have visited num_test_image test images then we should ignore test images later
            # then we should check args.num_part and args.part_id, if pass, we should go on to attack this image

            is_meet_each_type = OrderedDict([
                (key, (is_correct & is_image_type[key])[:-1].sum().item() >= num_image_each_type[key])
                for key in is_image_type.keys()
            ])
            if all(is_meet_each_type.values()):
                # attack task is done
                is_ignore[-1] = 1
                log.info('We have visited enough train_seen/train_unseen/val/test images, attack task is done')
                break
            if not is_correct[-1]:
                # if misclassified, we ignore directly
                is_ignore[-1] = 1
                correct_all.append(0)
                log.info('Ignore {}-th image: image_id: {}, since it is misclassified'.format(
                    batch_index, image_id.item()))

            else:
                correct_all.append(1)
                self.images_index = self.images_index + 1
                # current image is correctly classified
                assert is_image_type[image_type][-1].item()
                if is_meet_each_type[image_type]:
                    # we've seen enough that type of images, so we should ignore
                    is_ignore[-1] = 1
                    log.info('Ignore {}-th image: image_id: {}, since we have visited enough ({}) {} images'.format(
                        batch_index, image_id.item(), num_image_each_type[image_type], image_type))
                else:
                    # we've not visited enough that type of images, now we should check part related arguments
                    if args.num_part == 1:
                        # if there is only 1 part, we should go on to attack it
                        pass
                    else:
                        # goes here indicates is_meet_each_type[key] is False, so there must be at least 1 image
                        assert num_image_each_type[image_type] > 0
                        num_image_each_part = num_image_each_type[image_type] // args.num_part
                        assert num_image_each_part > 0
                        current_num = (is_correct & is_image_type[
                            image_type]).sum().item() - 1  # make index starts from 0
                        current_image_part_id = min(current_num // num_image_each_part, args.num_part - 1)
                        if current_image_part_id == args.part_id:
                            # current part matches args.part_id, we should run
                            pass
                        else:
                            is_ignore[-1] = 1
                            log.info(
                                'Ignore {}-th image: image_id: {}, since part of current image is {} '
                                '({} images for each part) while args.part_id is {}'.format(
                                    batch_index, image_id.item(), current_image_part_id,
                                    num_image_each_part, args.part_id))
            # ignore image
            if is_ignore[-1].item():
                continue
            # reset policy model and optimizer
            policy.reinit()
            if args.policy_arch.endswith('_inv'):
                policy.net.empty_coeff.data[:] = args.empty_coeff
            policy.zero_grad()
            optimizer.zero_grad()
            optimizer.load_state_dict(optimizer_init_state_dict)

            # reset grad estimation (only has effect if args.grad_method == 'momentum')
            grad = torch.zeros_like(image)
            # initialize last true grad, we will calculate the cos sim between true grads of consecutive iterations
            last_true_grad = torch.zeros_like(image)

            # reset model
            if args.targeted == 'untargeted':
                target = None
                victim.reset(image=image, label=label, target_label=None,
                             attack_type=args.targeted, norm_type=args.norm_type)
            # reset done, start attacking
            log.info(
                'Attacking {}-th image: image_id: {}, image_type: {}'.format(batch_index, image_id.item(), image_type))

            # init stat counters and some variables for current image
            query_count_all = torch.LongTensor(0)
            distance_all = torch.FloatTensor(0)
            sim_all = torch.FloatTensor(0)
            lr = args.lr
            init_boost = args.init_boost
            distance_decayed = None

            # find initial point using blended uniform noise
            # set random seed based on image_id, so we will have the same starting point for different attacking algorithms
            random.seed(image_id.item())
            np.random.seed(image_id.item())
            torch.manual_seed(image_id.item())
            torch.cuda.manual_seed(image_id.item())
            torch.cuda.manual_seed_all(image_id.item())

            # --[for attack]
            if args.targeted == 'targeted' and args.target_type == "load_random":
                loaded_target_labels = np.load("./target_class_labels/{}/label.npy".format(args.dataset))
                loaded_target_labels = torch.from_numpy(loaded_target_labels).long()
            selected = torch.arange(batch_index * args.batch_size,
                                    min((batch_index + 1) * args.batch_size, self.total_images))
            if args.targeted == 'targeted':
                if args.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[args.dataset],
                                                  size=label.size()).long()
                    invalid_target_index = target_labels.eq(label)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=logit.shape[1],
                                                                            size=target_labels[invalid_target_index].shape).long()
                        invalid_target_index = target_labels.eq(label)
                elif args.target_type == "load_random":
                    target_labels = loaded_target_labels[selected]
                    assert target_labels[0].item() != label[0].item()
                    # log.info("load random label as {}".format(target_labels))
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1).detach().cpu()
                elif args.target_type == "increment":
                    target_labels = torch.fmod(label + 1, CLASS_NUM[args.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(args.target_type))

                victim.reset(image=image, label=None, target_label=target_labels,
                             attack_type=args.targeted, norm_type=args.norm_type)
                target_images = self.get_image_of_target_class(self.dataset, target_labels)
                target = target_labels
                if target_images is None:
                    log.info("{}-th image cannot get a valid target class image to initialize!".format(batch_index + 1))
                    continue
            else:
                target_labels = None
                target_images = None
            init_adv_image, num_eval = self.initialize(image, target_images, label, target_labels)
            ##########################
            if init_adv_image is None:
                log.info('Initial point not found, {}-th image: image_id: {}, skip this image'.format(
                    batch_index, image_id.item()))
                continue
            distance_before_binary_search = self.calc_distance(image, init_adv_image).item()
            # clone init_adv_image since we want to save init_adv_image into pickle file later
            # note this has to be done before binary search, since if we save init_adv_image after binary search,
            # sometimes it would lead to ambiguity between the first and second largest logit
            # foolbox_attacks.py might think it's an invalid starting point and raise exceptions
            init_adv_image = victim.best_adv_image.clone()
            init_distance = victim.best_distance.clone()
            init_query_count = victim.query_count
            distortion_list.append(init_distance.item())
            query_list.append(init_query_count) # --[debug]
            # after cloning init_adv, init_distance, and init_query_count, we can do binary search
            distance_after_binary_search = self.calc_distance(image,
                                                              self.binary_search(victim, image, init_adv_image, theta)).item()
            last_save_distance = distance_after_binary_search
            log.info('Initial point found, {}-th image: image_id: {}, query: {}, '
                     'dist pre / post bs: {:.4f} / {:.4f}'.format(batch_index, image_id.item(), victim.query_count,
                                                                  distance_before_binary_search,
                                                                  distance_after_binary_search))

            # record query count and distance for initialization
            query_count_all = torch.cat((query_count_all, torch.LongTensor([victim.query_count])))
            distance_all = torch.cat((distance_all, torch.FloatTensor([victim.best_distance.item()])))
            sim_all = torch.cat((sim_all, torch.zeros(1)))

            # do pre-tune here to make sure ce / lce / tce are correct for inverse networks
            if not args.pre_tune_after_ib:
                do_pre_tune(adv_image_=victim.best_adv_image, image_=image, label_=label, target_=victim.best_adv_label)
            # start iterative attack
            remember_index = 0
            compare_distortion_query = 0
            compared = False
            for step_index in itertools.count():
                if victim.query_count >= args.max_query:
                    query_all.append(victim.query_count) # --[debug]
                    distortion_with_max_querys.append(victim.best_distance.item())
                    break

                # load newest best adv image & distance & query count
                adv_image = victim.best_adv_image
                if args.targeted == 'untargeted':
                    # use current adv label as target
                    target = victim.best_adv_label  # target for this round
                else:
                    # we've already initialized the target variable
                    pass
                assert target is not None
                last_distance = victim.best_distance.clone()
                last_query_count = copy.deepcopy(victim.query_count)

                # set init boost status
                if init_boost:
                    # check whether we should stop init boost
                    if args.grad_method != 'policy_distance':
                        log.info('Step {}, disable init boost since args.grad_method is {}'.format(
                            step_index + 1, args.grad_method))
                        init_boost = False
                    if init_boost and args.policy_arch == 'empty':
                        log.info('Step {}, disable init boost since policy architecture is {}'.format(
                            step_index + 1, args.policy_arch))
                        init_boost = False
                    if init_boost and victim.query_count >= args.init_boost_stop:
                        log.info('Step {}, disable init boost since current query count ({}) reaches limit {}'.format(
                            step_index + 1, victim.query_count, args.init_boost_stop))
                        init_boost = False
                    if init_boost and (distance_decayed is not None) and (distance_decayed < args.init_boost_th):
                        log.info('Step {}, disable init boost since distance decay in last iteration {:.4f} '
                                 'does not reach threshold {:.4f}'.format(step_index + 1,
                                                                          distance_decayed, args.init_boost_th))
                        init_boost = False

                    # if init boost is closed in this iteration and args.pre_tune_after_ib is True, i
                    # we should do pre-tune here to make sure ce / lce / tce are correct for inverse networks
                    if (not init_boost) and args.pre_tune_after_ib:
                        do_pre_tune(adv_image_=victim.best_adv_image, image_=image,
                                    label_=label, target_=target)

                    # if init boost is closed in this iteration, we should init empty_normal_mean
                    if args.init_empty_normal_mean and (not init_boost) and args.policy_arch.endswith('_inv'):
                        # initialize policy.net.empty_normal_mean
                        # we need to grab the newest mean vector from pre-trained weights
                        # another important thing is setting empty_normal_mean to correct norm, reason why norm matters:
                        # consider x1 = 1000 * x2, and we have h1 = x1 / norm(x1) and h2 = x2 / norm(x2)
                        # obviously h1 = h2. when the gradients backward through h to x, assume we have
                        # h1.grad = h2.grad, and thus x1.grad = 0.001 * x2.grad instead of x1.grad = 1000 * x2.grad!
                        # this phenomenon could be catastrophic since large vector (x1) has smaller gradient, which
                        # could effectively prevents it from been updated by the first-order gradient based optimizer.
                        # so we need to set it to a suitable norm when initializing it
                        with torch.no_grad():
                            empty_normal_mean = policy(adv_image, image, label, target,
                                                       output_fields=('grad',))['grad'].detach()
                        empty_normal_mean = empty_normal_mean / empty_normal_mean.norm() * args.empty_normal_mean_norm
                        empty_normal_mean = empty_normal_mean.view(*policy.net.empty_normal_mean.shape)
                        policy.net.empty_normal_mean.data[:] = empty_normal_mean
                        # args.grad_method = 'bapp'
                    else:
                        # we do not initialize policy.net.empty_normal_mean now
                        pass
                else:
                    # if init boost has already been stopped, we never re-open it
                    pass

                # cut learning rate every args.lr_step_freq steps by args.lr_step_mult
                if args.lr_step_mult != 1.0 and args.lr_step_freq > 0 and (step_index + 1) % args.lr_step_freq == 0:
                    for param_group in optimizer.param_groups:
                        lr = param_group['lr']
                        new_lr = lr * args.lr_step_mult
                        param_group['lr'] = new_lr
                        log.info('Step {}, cut learning rate from {:g} to {:g}'.format(step_index + 1, lr, new_lr))
                        lr = new_lr

                # record query count for each step
                query_count_current_step = list()

                # **** step 1: generate sampling points ****
                # choose number of evaluations
                if args.fix_num_eval:
                    num_eval = int(min(args.init_num_eval, args.max_num_eval))
                else:
                    num_eval = int(min([args.init_num_eval * math.sqrt(step_index + 1), args.max_num_eval]))

                if args.grad_method in ['bapp', 'momentum', 'true_grad']:
                    # generate noise to construct sampling points
                    if args.grad_size == 0:
                        noise_shape = (num_eval, image.shape[1], image.shape[2], image.shape[3])
                    else:
                        noise_shape = (num_eval, image.shape[1], args.grad_size, args.grad_size)
                    if args.norm_type == 'l2':
                        if args.use_pytorch_rng:
                            rv = torch.randn(*noise_shape).to(device)
                        else:
                            rv = np.random.randn(*noise_shape)
                            rv = torch.FloatTensor(rv).to(device)
                    elif args.norm_type == 'linf':
                        if args.use_pytorch_rng:
                            rv = (2 * torch.rand(*noise_shape) - 1).to(device)
                        else:
                            rv = np.random.uniform(low=-1, high=1, size=noise_shape)
                            rv = torch.FloatTensor(rv).to(device)
                    else:
                        raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                elif args.grad_method in ['policy_distance']:
                    # get mean/std using policy network

                    output = policy(adv_image, image, label, target, output_fields=output_fields)
                    mean, std = output['grad'], output['std']
                    if args.dataset != 'debug':
                        assert mean.dim() == 4 and mean.shape[0] == 1 and std.numel() == 1
                    else:
                        assert mean.dim() == 2 and mean.shape[0] == 1 and std.numel() == 1

                    # prevent mean to be too large
                    abs_mean = mean.abs().mean().item()
                    if abs_mean / std.item() > args.max_sharp:
                        policy.rescale(args.max_sharp * std.item() / abs_mean)
                        output = policy(adv_image, image, label, target, output_fields=output_fields)
                        mean, std = output['grad'], output['std']

                    # if args.init_boost is True, we might skip step 3, so we compute ce / lce / tce here
                    if args.policy_arch.endswith('_inv'):
                        # logit for adv_image
                        adv_logit = output['adv_logit']
                        lce = F.cross_entropy(adv_logit, label.view(-1), reduction='none')
                        tce = F.cross_entropy(adv_logit, target.view(-1), reduction='none')

                        # logit for clean_image
                        logit = output['logit']
                        ce = F.cross_entropy(logit, label.view(-1), reduction='none')

                    # bundle mean / std into distribution
                    distribution = Normal(mean.view(*mean.shape[1:]), std)
                    mean = mean.detach()

                    # sample actions from generated distribution
                    action = distribution.sample((num_eval,))

                    # rv is used in step 2 and 3 for all methods
                    rv = action.clone()
                else:
                    raise ValueError('Unknown grad method: {}'.format(args.grad_method))

                query_count_current_step.append(victim.query_count - last_query_count - sum(query_count_current_step))
                query_list.append(victim.query_count)
                distortion_list.append(victim.best_distance.item()) # --[debug]
                if victim.best_distance.item() <= self.epsilon and not compared:
                    compare_distortion_query = victim.query_count
                    compared = True
                # **** step 2: query generated sampling points to collect information about the decision boundary ****
                rv = upsampler(rv)
                rv = rv / self.norm(rv)

                # choose delta
                if step_index == 0:
                    delta = 0.1
                    baseline = 0.05
                else:
                    if args.norm_type == 'l2':
                        delta = math.sqrt(d) * theta * distance_before_binary_search * args.delta_mult
                    elif args.norm_type == 'linf':
                        delta = d * theta * distance_before_binary_search * args.delta_mult
                    else:
                        raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))

                # choose epsilon
                if args.epsilon_schema == 'fixed':
                    init_epsilon = distance_after_binary_search * args.fixed_epsilon_mult
                elif args.epsilon_schema == 'sqrt':
                    init_epsilon = distance_after_binary_search / math.sqrt(step_index + 1)
                elif args.epsilon_schema == 'cosine':
                    period = int(args.max_query / args.cosine_epsilon_round)
                    cosine_val = (math.cos(math.pi * (victim.query_count % period) / float(period)) + 1) / 2
                    init_epsilon = distance_after_binary_search * cosine_val
                else:
                    raise ValueError('Unknown epsilon schema: {}'.format(args.epsilon_schema))

                if args.grad_method in ['bapp', 'momentum', 'true_grad']:
                    adv_image_perturbed = torch.clamp(adv_image + delta * rv, 0, 1)
                    rv = (adv_image_perturbed - adv_image) / delta
                    # query the model using sampled inputs
                    with torch.no_grad():
                        decisions = self.decision_function(victim, adv_image_perturbed)
                    fval = 2 * decisions.float() - 1.0
                elif args.grad_method in ['policy_distance']:
                    # for policy_distance, we query sampling points in reward assignment step
                    pass
                else:
                    raise ValueError('Unknown grad method: {}'.format(args.grad_method))

                query_count_current_step.append(victim.query_count - last_query_count - sum(query_count_current_step))
                query_list.append(victim.query_count)
                distortion_list.append(victim.best_distance.item())  # --[debug]
                if victim.best_distance.item() <= self.epsilon and not compared:
                    compare_distortion_query = victim.query_count
                    compared = True
                # **** step 3: update gradient estimation or policy network using collected information ****
                # we will print grad_norm = 0 in log if we do not calculate this variable later
                grad_norm = 0
                if args.grad_method in ['bapp', 'momentum']:
                    # use bapp to estimate gradient
                    # baseline subtraction (when fval differs)
                    fval = fval.view(-1, 1, 1, 1)
                    if args.sub_base:
                        if fval.mean().item() == 1.0:
                            gradt = rv.mean(dim=0)
                        elif fval.mean().item() == -1.0:
                            gradt = -rv.mean(dim=0)
                        else:
                            gradt = ((fval - fval.mean()) * rv).mean(dim=0)
                    else:
                        gradt = (fval * rv).mean(dim=0)
                    gradt = gradt.view(*image.shape)
                    if args.grad_method == 'bapp':
                        grad = gradt
                    elif args.grad_method == 'momentum':
                        grad = gradt + args.lr * grad / self.norm(grad, p=1)
                    else:
                        raise ValueError('Grad method should be bapp or momentum')
                    grad_for_save = grad.clone()
                elif args.grad_method == 'true_grad':
                    with torch.enable_grad():
                        adv_image.requires_grad = True
                        if adv_image.grad is not None:
                            adv_image.grad[:] = 0.
                        logit = victim(adv_image)
                        assert not logit.argmax().eq(label).item()
                        loss = self.cw_loss(logit, label, target)
                        grad = torch.autograd.grad(loss, adv_image)[0]
                        assert grad.shape == adv_image.shape
                        adv_image.requires_grad = False
                    grad_for_save = grad.clone()
                elif args.grad_method in ['policy_distance']:
                    # calculate next_state and reward, and store (s, a, r, s)
                    if not init_boost:
                        # next state is not need for single-step REINFORCE
                        # TODO: set next_state for multi-step RL algorithms
                        next_state = torch.zeros_like(action)

                        # perturb image along actions for epsilon distance
                        if args.norm_type == 'l2':
                            rv = rv / self.norm(rv)
                        elif args.norm_type == 'linf':
                            rv = torch.sign(rv)
                        else:
                            raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                        adv_image_perturbed = torch.clamp(adv_image + delta * rv, 0, 1)

                        # use current mean distance decay to decide baseline
                        if len(args.try_split) == 1 and args.try_split[0] == 0.0:
                            # since there is only one level (0), so we do not need to evaluate the value of baseline
                            pass
                        else:
                            mean = policy(victim.best_adv_image, image, label, victim.best_adv_label,
                                          output_fields=('grad',))['grad']
                            mean = upsampler(mean)
                            if mean.norm().item() > 0 and num_eval > 1:
                                # we use mean vector to decide the baseline
                                if args.norm_type == 'l2':
                                    t = torch.clamp(adv_image + delta * mean / self.norm(mean), 0, 1)
                                elif args.norm_type == 'linf':
                                    t = torch.clamp(adv_image + delta * torch.sign(mean), 0, 1)
                                else:
                                    raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                                if self.decision_function(victim, t).item():
                                    # binary search is valid if t is adversarial
                                    t = self.binary_search(victim, image, t, theta * args.current_mean_mult)
                                    t = self.calc_distance(image, t)
                                    if (t > last_distance).item():
                                        # we get larger distance after jump and binary search
                                        # so we will rescale baseline
                                        baseline *= args.rescale_factor
                                    else:
                                        baseline = (last_distance - t).item()
                                else:
                                    # binary search is invalid since two points are both un-adversarial
                                    # so we will rescale baseline
                                    baseline *= args.rescale_factor
                            else:
                                # this should happen only in the first iteration and policy arch is empty
                                # and we have already assigned value for baseline in this case
                                pass
                        baseline = max(baseline, args.min_baseline * delta)
                        baseline = min(baseline, args.max_baseline * delta)

                        # perform reward assignment based on try split multipliers and baseline
                        reward = torch.zeros(num_eval).to(device)
                        unassigned = torch.ones(num_eval).byte().to(device)
                        for i in itertools.count():
                            if i < len(args.try_split):
                                multiplier = args.try_split[i]
                            else:
                                multiplier = args.try_split[-1] + 0.25 * (i - len(args.try_split) + 1)

                            # calculate try distance using baseline and multiplier
                            try_distance = last_distance - multiplier * baseline
                            if args.try_aggressive:
                                try_distance -= distance_before_binary_search * theta

                            # shrinkage the perturbed image according to try_distance
                            # and then check whether or not the shrinkage still satisfies adversarial criterion
                            # if the adversarial criterion still holds, we think the corresponding action has good
                            # potential and thus we assign a higher reward to it
                            v = adv_image_perturbed[unassigned] - image
                            if args.norm_type == 'l2':
                                v = v / self.norm(v)
                            elif args.norm_type == 'linf':
                                v = torch.sign(v)
                            else:
                                raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                            decisions = self.decision_function(victim, torch.clamp(image + try_distance * v, 0, 1))
                            decisions = decisions.byte()
                            t = torch.zeros(num_eval).byte().to(device)
                            t[unassigned] = decisions
                            reward[unassigned & (~t)] = i
                            unassigned = unassigned & t

                            # break condition: match anyone will cause break
                            #   1. if all actions have been assigned, break
                            #   2. condition 1 False, args.try_split done, args.force_diverse_reward is False
                            #   3. condition 1 False, args.try_split done, args.force_diverse_reward is True, \
                            #        and we've at least assigned one action
                            if not unassigned.any().item():
                                # condition 1
                                break
                            elif i >= len(args.try_split) - 1:
                                if not args.force_diverse_reward:
                                    # condition 2
                                    break
                                elif (~unassigned).any().item():
                                    # condition 3
                                    break
                            else:
                                # otherwise, continue to try more multipliers
                                pass

                        # assign rewards for remaining actions
                        reward[unassigned] = i + 1

                        # perform resampling
                        if args.resample:
                            resample_indices = list()
                            success = (reward > 0).nonzero()[:, 0]
                            if success.numel() > 0:
                                resample_indices += success.tolist()
                            fail = (reward <= 0).nonzero()[:, 0]
                            if fail.numel() > 0:
                                # shuffle fail first
                                fail = fail[torch.randperm(fail.numel())]

                                # then add at least one
                                resample_indices += fail.tolist()[:max(2, success.numel())]
                            resample_indices = torch.LongTensor(resample_indices)
                            rv = rv[resample_indices]
                            action = action[resample_indices]
                            next_state = next_state[resample_indices]
                            reward = reward[resample_indices]

                        # perform normalization on reward, hope that RL can learn faster
                        raw_reward = reward.clone()  # raw_reward is used to display reward information before modification
                        if args.exp_reward:
                            reward = 2 ** reward
                        if args.mean_reward:
                            reward = reward - reward.mean()
                        if args.std_reward:
                            if reward.std().item() > 0:
                                reward = reward / torch.clamp(reward.std(), min=1e-5)

                        # minus ca sim
                        if args.minus_ca_sim > 0:
                            ca = adv_image - image
                            ca_sim = (rv * ca).view(rv.shape[0], -1).sum(dim=1) / \
                                     ca.view(ca.shape[0], -1).norm(dim=1) / rv.view(rv.shape[0], -1).norm(dim=1)
                            reward = reward - ca_sim.abs() * args.minus_ca_sim

                        # one-step REINFORCE
                        action_log_prob = distribution.log_prob(action)
                        reward_loss = -reward.view(reward.shape[0], 1, 1, 1).detach() * action_log_prob
                        if args.policy_arch.endswith('_inv'):
                            # combine and make loss
                            loss = None
                            for lmbd, l in zip(args.ce_lmbd, [reward_loss, ce, lce, tce]):
                                if lmbd != 0:
                                    if loss is None:
                                        loss = lmbd * l
                                    else:
                                        loss = loss + lmbd * l
                            assert loss is not None, 'args.ce_lmbd must have at least 1 non-zero term'
                        else:
                            loss = reward_loss

                        optimizer.zero_grad()
                        policy.zero_grad()
                        loss.mean().backward()

                        # save clip grad norm for logging
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            functools.reduce(lambda g1, g2: g1['params'] + g2['params'], param_groups), args.clip_grad,
                            norm_type=2)

                        # detect whether or not nan appears in any parameter
                        for param_group in optimizer.param_groups:
                            for p in param_group['params']:
                                if torch.isnan(p).any().item():
                                    log.info('Found nan in policy network, exit')
                                    exit(1)

                        # update parameters
                        optimizer.step()

                        # reward.min() == reward.max() indicates the sampling process is failed
                        # thus we need to shrinkage policy network if mean is too large
                        if raw_reward.min().item() == raw_reward.max().item() == 0 and \
                                mean.abs().mean().item() > std.item() * args.min_sharp:
                            policy.rescale(args.rescale_factor)
                    else:
                        # in init boost status, we should set raw_reward and reward for logging
                        raw_reward = reward = torch.zeros(num_eval).to(device)

                    # get gradient estimation using updated policy network
                    # we will use this to update adv_image in step 4
                    grad_for_save = policy(adv_image, image, label, target, output_fields=('grad',))['grad'].detach()
                    grad_for_save_np = grad_for_save.cpu().detach().numpy()
                    # grad_for_save_np = grad_for_save_np.transpose((1, 2, 3, 0))  # 将维度转换成 (3, 32, 32, 1)
                    grad_for_save_np = grad_for_save_np.squeeze()
                    grad = upsampler(grad_for_save.clone())
                else:
                    raise ValueError('Unknown grad method: {}'.format(args.grad_method))

                query_count_current_step.append(victim.query_count - last_query_count - sum(query_count_current_step))

                query_list.append(victim.query_count)
                distortion_list.append(victim.best_distance.item())  # --[debug]
                if victim.best_distance.item() <= self.epsilon and not compared:
                    compare_distortion_query = victim.query_count
                    compared = True
                # step 4: exploit learned information to find better adv_image: jump and then binary search

                # get cosine similarity of relevant vectors (just for debug, so we exclude these query counts)
                grad = grad.detach()
                grad = grad / self.norm(grad)
                with torch.enable_grad():
                    adv_image.requires_grad = True
                    if adv_image.grad is not None:
                        adv_image.grad[:] = 0.
                    logit = victim(adv_image, no_count=True)
                    assert not logit.argmax().eq(label).item()
                    loss = self.cw_loss(logit, label, target)
                    true_grad = torch.autograd.grad(loss, adv_image)[0]
                    adv_image.requires_grad = False
                assert true_grad.shape == grad.shape == adv_image.shape
                sim = (grad * true_grad).sum() / grad.norm() / true_grad.norm()
                ca = adv_image - image
                ca_sim = (grad * ca).sum() / grad.norm() / ca.norm()
                sim_all = torch.cat((sim_all, torch.FloatTensor([sim.item()])))
                last_true_grad_sim = (last_true_grad * true_grad).sum() / last_true_grad.norm() / true_grad.norm()
                last_true_grad = true_grad.clone()

                # perform jump and do binary search for multiple times using the same grad vector
                # we need to save grad before entering the loop, since we might modify the grad in the loop later
                grad = grad / self.norm(grad)  # for linf case, perform a l2 normalization does not matter
                grad_clone = grad.clone()
                used_epsilon = list()
                able_to_jump = list()
                able_to_decrease = list()
                pre = list()
                post = list()

                # we only jump once if init boost is activated, since in initial stage epsilon is very large, and
                # jump once for each init_epsilon can yield better performance
                for jump_index in range(args.jump_count):
                    # save last jump distance to determine whether or not current jump leads to distortion reduction
                    if jump_index == 0:
                        # we use last_distance instead of model.best_distance for the first jump trial
                        # because the gradient is estimated at adv_image instead of model.best_adv_image
                        # the sampling process in step 3 might potentially modify model.best_adv_image
                        last_jump_distance = last_distance
                    else:
                        # for later jump trials, we simply use gradient estimated at adv_image on model.best_adv_image
                        last_jump_distance = victim.best_distance

                    # geometric progression for suitable epsilon (i.e., jump step size)
                    epsilon = init_epsilon
                    able_to_jump.append(False)
                    able_to_decrease.append(False)
                    while epsilon > init_epsilon * args.min_epsilon_mult:
                        # perturb the adv image by adding the grad first
                        grad = grad_clone.clone()  # grad_clone has been already normalized
                        if jump_index == 0:
                            # for the first jump trial we use adv_image since the gradient is estimated at it
                            jump_center = adv_image
                        else:
                            # for later jump trials we use the newest adv image
                            jump_center = victim.best_adv_image

                        # grab the grad vector
                        if args.calibrate_clip:
                            # we calibrate the grad to alleviate te clipping effect
                            # this could improve the utility of the perturbation budget
                            adv_image_perturbed = torch.clamp(jump_center + epsilon * grad, 0, 1)
                            grad = (adv_image_perturbed - jump_center) / epsilon
                            grad = grad / self.norm(grad)

                        # update the image
                        if args.norm_type == 'l2':
                            grad = grad / self.norm(grad)
                        elif args.norm_type == 'linf':
                            grad = torch.sign(grad)
                        else:
                            raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                        adv_image_perturbed = torch.clamp(jump_center + epsilon * grad, 0, 1)

                        # there are two different ways to exploit the jump direction:
                        #  1. (default) use bapp-style jump: adv_image -> add grad -> binary search
                        #  2. use tangent-style jump: adv_image -> add grad -> project onto ball -> binary search
                        # the method 2 inherits the same idea from reward assignment
                        if args.tan_jump:
                            # if args.tan_jump is set to True, we use tangent-style jump
                            # i.e., project perturbed image onto a ball centered at image with radius last_jump_distance
                            v = adv_image_perturbed - image
                            if args.norm_type == 'l2':
                                v = v / self.norm(v)
                            elif args.norm_type == 'linf':
                                v = torch.sign(v)
                            else:
                                raise NotImplementedError('Unknown norm: {}'.format(args.norm_type))
                            adv_image_perturbed = torch.clamp(image + last_jump_distance * v, 0, 1)

                        # if after jumping the perturbed image lies in the adversarial region, we can do binary search
                        if self.decision_function(victim, adv_image_perturbed).item():
                            able_to_jump[-1] = True
                            break

                        # if not break (the while loop), then we need to try smaller epsilon
                        epsilon /= 2.0

                    # append final epsilon for this round
                    used_epsilon.append(epsilon)

                    # binary search to project the point to the decision boundary
                    if able_to_jump[-1]:
                        assert adv_image_perturbed.shape == image.shape
                        distance_before_binary_search = self.calc_distance(image, adv_image_perturbed).item()
                        adv_image_searched = self.binary_search(victim, image, adv_image_perturbed, theta)
                        distance_after_binary_search = self.calc_distance(image, adv_image_searched).item()
                        able_to_decrease[-1] = distance_after_binary_search < last_jump_distance.item()
                        pre.append(distance_before_binary_search)
                        post.append(distance_after_binary_search)

                    # if this round cannot decrease distance, using the same grad for the next round would be meaning less
                    if not able_to_decrease[-1]:
                        break

                # recover grad from grad_clone, and we might use it to calculate something for logs
                grad = grad_clone.clone()

                if args.grad_method == 'policy_distance':
                    if able_to_jump[0] is False and mean.abs().mean().item() > std.item() * args.min_sharp:
                        policy.rescale(args.rescale_factor)

                query_count_current_step.append(victim.query_count - last_query_count - sum(query_count_current_step))
                query_list.append(victim.query_count)
                distortion_list.append(victim.best_distance.item()) # --[debug]
                if victim.best_distance.item() <= self.epsilon and not compared:
                    compare_distortion_query = victim.query_count
                    compared = True
                # record query count and distance
                query_count_all = torch.cat((query_count_all, torch.LongTensor([victim.query_count])))
                distance_all = torch.cat((distance_all, torch.FloatTensor([victim.best_distance.item()])))

                # how much distance is decayed in this iteration
                distance_decayed = (last_distance - victim.best_distance).item()
                distance_decayed_pct = distance_decayed / last_distance.item() * 100
                last_save_pct = (victim.best_distance / last_save_distance).item() * 100

                # save gradient
                if args.save_grad:
                    # only save if distance decay enough in this iteration
                    if last_save_pct <= 100 - args.save_grad_pct:
                        # save grads
                        path_for_grad = osp.join(args.save_grad_dir, '{}'.format(args.dataset),
                                                 '{}'.format(args.victim_arch), 'grad_save.npy')
                        path_for_true_grad = osp.join(args.save_grad_dir, '{}'.format(args.dataset),
                                                      '{}'.format(args.victim_arch), 'true_grad.npy')
                        path_for_adv_image = osp.join(args.save_grad_dir, '{}'.format(args.dataset),
                                                      '{}'.format(args.victim_arch), 'adv_iamge.npy')
                        true_grad_for_save = true_grad.clone()
                        true_grad_for_save = true_grad_for_save.cpu().detach().numpy()
                        # true_grad_for_save = true_grad_for_save.transpose((1, 2, 3, 0))
                        true_grad_for_save = true_grad_for_save.squeeze()
                        adv_image_for_save = adv_image.clone()
                        adv_image_for_save = adv_image_for_save.cpu().detach().numpy()
                        # adv_image_for_save = adv_image_for_save.transpose((1, 2, 3, 0))
                        adv_image_for_save = adv_image_for_save.squeeze()
                        if num_of_index + step_index == 0:
                            fp = np.memmap(path_for_grad, dtype=np.float32, mode='w+',
                                           shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp[:] = grad_for_save_np
                            # fp.flush()
                            del fp
                            fp1 = np.memmap(path_for_true_grad, dtype=np.float32, mode='w+',
                                            shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp1[:] = true_grad_for_save
                            # fp1.flush()
                            del fp1
                            fp2 = np.memmap(path_for_adv_image, dtype=np.float32, mode='w+',
                                            shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp2[:] = adv_image_for_save
                            # fp2.flush()
                            del fp2
                        else:
                            fp = np.memmap(path_for_grad, dtype=np.float32, mode='r+',
                                           shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp[:] = grad_for_save_np
                            # fp.flush()
                            del fp
                            fp1 = np.memmap(path_for_true_grad, dtype=np.float32, mode='r+',
                                            shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp1[:] = true_grad_for_save
                            # fp1.flush()
                            del fp1
                            fp2 = np.memmap(path_for_adv_image, dtype=np.float32, mode='r+',
                                            shape=(channel, size_of_image, size_of_image), offset=(self.num_of_step) * channel * size_of_image * size_of_image * 32 // 8)
                            fp2[:] = adv_image_for_save
                            # fp2.flush()
                            del fp2
                        last_save_distance = victim.best_distance.item()
                        self.num_of_step = self.num_of_step + 1

                # print("distortion" + str(victim.best_distance.item()))
                # print('query count: {:d}'.format(victim.query_count))
                if args.reset_each_step:
                    policy.reinit()
                remember_index = step_index
                # print message
                log.info('Attacking {}-th image, image_id {}, label {}, target {}, image_type {}, step {}'.format(
                    batch_index, image_id.item(), label.item(), target.item(), image_type, step_index + 1))
                log.info('   distance / decay: {:.4f} ({:.2f}% last save) / {:.4g} ({:.2f}% last iteration)'.format(
                    victim.best_distance, last_save_pct, distance_decayed, distance_decayed_pct))
                log.info('        query count: {:d}'.format(victim.query_count))
                log.info(' query count detail: {}'.format(query_count_current_step))
                log.info('  num eval / reward: {:d} / {:d}'.format(num_eval, rv.shape[0]))
                if args.grad_method in ['bapp', 'momentum', 'true_grad']:
                    log.info('          eval mean: {:.4g}'.format(fval.mean()))
                if args.grad_method in ['policy_distance']:
                    log.info('   reward[0] / hist: {} / {}'.format(
                        raw_reward[0].item(), collections.Counter(raw_reward.tolist())))
                    log.info(' reward min/med/max: {:.4g} / {:.4g} / {:.4g}'.format(
                        reward.min(), reward.median(), reward.max()))

                    output = policy(victim.best_adv_image, image, label, victim.best_adv_label,
                                    output_fields=('grad', 'std'))

                    adv_image = F.interpolate(adv_image,
                                              size=(299, 299),
                                              mode='bilinear', align_corners=False)
                    mean, std = output['grad'], output['std']
                    log.info('       |mean| / std: {:.4g} / {:.4g} = {:.4g}'.format(
                        mean.abs().mean(), std.mean(), mean.abs().mean() / std.mean()))
                    log.info('             factor: {:.4g}'.format(policy.factor))
                    log.info('          lr / grad: {:.4g} / {:.4g}'.format(lr, grad_norm))
                    log.info('   baseline / delta: {:.4g} / {:.4g} = {:.4f}'.format(baseline, delta, baseline / delta))
                    log.info('         init boost: {}'.format(init_boost))
                    if args.policy_arch.endswith('_inv'):
                        log.info(
                            '     ce / lce / tce: {:.4g} / {:.4g} / {:.4g}'.format(ce.item(), lce.item(), tce.item()))
                log.info('       sim / ca sim: {:.4f} / {:.4f}'.format(sim, ca_sim))
                log.info(' last true grad sim: {:.4f}'.format(last_true_grad_sim))
                log.info('              delta: {:.4f}'.format(delta))
                log.info('    init eps / used: {:.4f} / {}'.format(init_epsilon, self.format_float_list(used_epsilon)))
                log.info('       able to jump: {}'.format(able_to_jump))
                log.info('   able to decrease: {}'.format(able_to_decrease))
                log.info(' dist pre / post bs: {} / {}'.format(self.format_float_list(pre), self.format_float_list(post)))

                # reset policy model after each step, usually for debug
            # --[debug]
            if compared:
                success_query_all.append(compare_distortion_query)
            else:
                success_query_all.append(args.max_query)
            for e in range(len(query_list)):
                key = str(query_list[e])
                distortion[key] = distortion_list[e]
            distortionsortList = []
            distortion_after_sort = {}
            for k in distortion.keys():
                distortionsortList.append([k, distortion[k]])
            afterSortlist = self.bubble_sort(distortionsortList)
            for e in afterSortlist:
                distortion_after_sort[e[0]] = e[1]
            alld[str(batch_index)] = distortion_after_sort
            if label == victim.best_adv_label:
                success_all.append(0)
                notdone_all.append(1)
            else:
                success_all.append(1)
                notdone_all.append(0)
            # --[debug]
            if args.save_grad:

                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'image_ids_step.json'), 'r') as f:
                    data = json.load(f)
                # 追加新的数据到JSON数据中
                # new_data = {str(image_id): er+1}
                data[str(image_id.item())] = remember_index + 1
                # 将更新的JSON数据写回到文件中
                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'image_ids_step.json'), 'w') as f:
                    json.dump(data, f)

                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'images_and_labels.json'), 'r') as f:
                    data_for_labels = json.load(f)
                data_for_labels[str(image_id.item())] = label.item()
                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'images_and_labels.json'), 'w') as f:
                    json.dump(data_for_labels, f)
                ######################
                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'image_ids_index.json'), 'r') as f:
                    data_for_index = json.load(f)
                data_for_index[str(image_id.item())] = self.images_index - 1
                with open(osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),
                                   'image_ids_index.json'), 'w') as f:
                    json.dump(data_for_index, f)
                ######################
                path_for_labels = osp.join(args.save_grad_dir, '{}'.format(args.dataset), '{}'.format(args.victim_arch),'image.npy')
                image_for_save = victim.image.cpu().detach().numpy()
                # image_for_save = image_for_save.transpose((1, 2, 3, 0))
                image_for_save = image_for_save.squeeze()
                if num_of_index == 0:
                    fp3 = np.memmap(path_for_labels, dtype=np.float32, mode='w+',
                                    shape=(channel, size_of_image, size_of_image),
                                    offset=0 * channel * size_of_image * size_of_image * 32 // 8)
                    fp3[:] = image_for_save
                    del fp3
                else:
                    fp3 = np.memmap(path_for_labels, dtype=np.float32, mode='r+',
                                    shape=(channel, size_of_image, size_of_image),
                                    offset=num_of_image_circle * channel * size_of_image * size_of_image * 32 // 8)
                    fp3[:] = image_for_save
                    del fp3
            num_of_image_circle = num_of_image_circle + 1
            # save result for current image
            # print progress
            print_progress('Up to now:')
            if num_of_index == 0:
                num_of_index = num_of_index + remember_index
            else:
                num_of_index = num_of_index + remember_index + 1
            # print progress
            print_progress('Up to now:')

        correct_avg = sum(correct_all) / len(correct_all)
        success_avg = sum(success_all) / (len(success_all) + 1)
        notdone_avg = sum(notdone_all) / (len(notdone_all) + 1)
        mean_query = sum(success_query_all) / (len(success_query_all) + 1)
        max_query = max(success_query_all)
        median_query = np.array(success_query_all)
        median_query = np.median(median_query)
        result_dump_path = osp.join(args.exp_dir, 'PDA_{}_{}'.format(args.dataset, args.norm_type), 'PDA.json')
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": correct_avg,
                          "avg_not_done": notdone_avg,
                          "mean_query": mean_query,
                          "median_query": median_query,
                          "max_query": max_query,
                          "correct_all": correct_all,
                          "not_done_all": notdone_all,
                          "success_all": success_all,
                          "query_all": query_all,
                          "success_query_all": success_query_all,
                          "distortion": alld,
                          "distortion_with_max_querys": distortion_with_max_querys,
                          "args": vars(args)}
        path = args.exp_dir + "/{}_result.json".format(self.victim_arch)
        with open(path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, indent=2)
        log.info("done, write stats info to {}".format(result_dump_path))
        # finished, create empty file thus others could check whether or not this task is done
        open(osp.join(args.exp_dir, 'done'), 'a').close()

        # print finish information
        log.info('Attack finished.')

    def format_float_list(self, array):
        return '[' + ', '.join(map(lambda t: '{:.4f}'.format(t), array)) + ']'


def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        # please note in this case, we need to import glog/logging after calling set_log_file(*, file_only=True)
        sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--dataset', required=True, type=str,
                        choices=['debug', 'CIFAR-100', 'MNIST', 'CIFAR-10', 'ImageNet', 'TinyImageNet'],
                        help='which dataset to be used')
    parser.add_argument('--phase', default='test', type=str, choices=['train', 'val', 'valv2', 'test'],
                        help='train, val, test')  # used
    parser.add_argument('--num-image', default=1000, type=int,
                        help='number of images to attack')
    parser.add_argument('--image-id-ref', default='', type=str,
                        help='reference image id sets')
    parser.add_argument('--part-id', default=0, type=int,
                        help='args.part_id is the id of current part among all args.num_part')
    parser.add_argument('--num-part', default=1, type=int,
                        help='the task could be split in several parts, args.num_part is the total number of parts')
    parser.add_argument('--victim-arch', default=None, type=str,
                        help='victim network architecture')
    parser.add_argument('--policy-arch', default='empty', type=str,
                        choices=['empty', 'vgg11_inv', 'vgg13_inv', 'vgg16_inv', 'vgg19_inv', 'carlinet_inv',
                                 'wrn_28_10_drop_inv', 'unet', 'resnet20_inv', 'resnet32_inv'],
                        help='policy network architecture, '
                             'pass ''empty'' for generating the gradient dataset before pretraining')
    parser.add_argument('--policy-weight-fname',
                        required=True, type=str,
                        help='pre-trained policy weight filename, eg., train_pytorch_model/policy_driven_attack/pretrained_models/CIFAR-10/vgg13_inv_WRN-28-10-drop/model_best.pth')
    parser.add_argument('--policy-bilinear', action='store_true',
                        help='use bilinear in policy network if applicable')
    parser.add_argument('--policy-normalization-type', default='none', type=str, choices=['none', 'bn', 'gn'],
                        help='normalization type in policy network if applicable')
    parser.add_argument('--policy-use-tanh', action='store_true',
                        help='use tanh in policy network if applicable')
    parser.add_argument('--policy-base-width', default=32, type=int,
                        help='set base width parameter in policy network if applicable')
    parser.add_argument('--policy-calibrate', action='store_true',
                        help='calibrate output of policy network using mean in policy network if applicable')
    parser.add_argument('--policy-init-std', default=0.003, type=float,
                        help='initial value of std for policy network')
    parser.add_argument('--empty-coeff', default=0.5, type=float,
                        help='to balance pre-trained weights and normal mean')
    parser.add_argument('--empty-normal-mean-norm', default=0.003, type=float,
                        help='set policy.net.empty_normal_mean.norm() when initialize it')
    parser.add_argument('--init-empty-normal-mean', action='store_true',
                        help='initial policy.net.empty_normal_mean using mean vector from the last iteration')
    parser.add_argument('--max-sharp', default=0.5, type=float,
                        help='maximal allowed sharp value, i.e., mean.abs().mean() / std')
    parser.add_argument('--min-sharp', default=0.02, type=float,
                        help='minimal allowed sharp value, i.e., mean.abs().mean() / std')
    parser.add_argument('--grad-method', default='policy_distance', type=str,
                        choices=['bapp', 'true_grad', 'policy_distance', 'momentum'],
                        help='gradient estimation method')
    parser.add_argument('--grad-size', default=0, type=int,
                        help='force to use a specific shape for grad')
    parser.add_argument('--sub-base', action='store_true',
                        help='subtract baseline for bapp and momentum gradient estimation')
    parser.add_argument('--try-split', default=[0.0, 0.25], nargs='+',
                        help='trying distance multipliers')
    parser.add_argument('--try-aggressive', action='store_true',
                        help='minus theta * l when trying distance multipliers')
    parser.add_argument('--force-diverse-reward', action='store_true',
                        help='if True, we will use more multipliers if '
                             'args.try_split is finished and all rewards are the same')
    parser.add_argument('--current-mean-mult', default=1.0, type=float,
                        help='to reduce binary search query counts when determining baseline')
    parser.add_argument('--exp-reward', action='store_true',
                        help='exponent (powered by 2) reward before training')
    parser.add_argument('--mean-reward', action='store_true',
                        help='subtract mean from reward before training')
    parser.add_argument('--std-reward', action='store_true',
                        help='divide reward by std of reward before training')
    parser.add_argument('--minus-ca-sim', default=0.0, type=float,
                        help='minus action cosine similarity with ca in reward')
    parser.add_argument('--resample', action='store_true',
                        help='resample actions to make success/fail roughly have the same weight')
    parser.add_argument('--rescale-factor', default=0.5, type=float,
                        help='factor to decrease mean_baseline and rescale policy network if no good sample is found')
    parser.add_argument('--min-baseline', default=0.05, type=float,
                        help='minimal allowed value for baseline')
    parser.add_argument('--max-baseline', default=1.0, type=float,
                        help='maximal allowed value for baseline')
    parser.add_argument('--min-epsilon-mult', default=0.01, type=float,
                        help='minimal allowed epsilon / init_epsilon')
    parser.add_argument('--reset-each-step', action='store_true',
                        help='reset policy model after each step')
    parser.add_argument('--num-pre-tune-step', default=0, type=int,
                        help='number of pre-tune steps to apply before starting to attack')
    parser.add_argument('--pre-tune-lr', default=0.0001, type=float,
                        help='pre-tune learning rate')
    parser.add_argument('--pre-tune-th', default=1.0, type=float,
                        help='stop pre-tune if max(ce, lce, tce) is smaller than this')
    parser.add_argument('--pre-tune-after-ib', action='store_true',
                        help='do pre-tune after init boost')
    parser.add_argument('--init-boost', action='store_true',
                        help='in first few round, do not perform sampling, just jump and search')
    parser.add_argument('--init-boost-stop', default=250, type=int,
                        help='stop init boost if query count reaches args.init_boost_stop')
    parser.add_argument('--init-boost-th', default=0.1, type=float,
                        help='stop init boost if distance decay in last iteration < args.init_boost_th')
    parser.add_argument('--jump-count', default=1, type=int,
                        help='in each iteration we do jump multiple times')
    parser.add_argument('--tan-jump', action='store_true',
                        help='use tangent-style jump instead of bapp-style')
    parser.add_argument('--ce-lmbd', nargs=4, default=[1.0, 3e-3, 1.5e-3, 1.5e-3],
                        help='cross entropy coefficient in loss, only applied if policy net could output logit')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='type of optimizer')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate') # used
    parser.add_argument('--lr-step-freq', default=0, type=int,
                        help='cut learning rate each args.lr_step_freq epoch')
    parser.add_argument('--lr-step-mult', default=1.0, type=float,
                        help='cut learning rate by args.lr_step_mult')
    parser.add_argument('--std-lr-mult', default=1.0, type=float,
                        help='std learning rate multiplier')
    parser.add_argument('--decay', default=1e-4, type=float,
                        help='weight decay') # used
    parser.add_argument('--momentum', default=0.0, type=float,
                        help='momentum') # used
    parser.add_argument('--clip-grad', default=5., type=float,
                        help='max gradient norm')
    parser.add_argument('--exclude-std', action='store_true',
                        help='do not optimize std')
    parser.add_argument('--targeted', default='untargeted', choices=['untargeted', 'targeted'],
                        help='type of attack, could be targeted or untargeted')
    parser.add_argument('--external-init-adv-image', action='store_true',
                        help='load external file for finding starting point')
    parser.add_argument('--norm-type', required=True, type=str, choices=['l2', 'linf'],
                        help='l2 attack or linf attack')
    parser.add_argument('--victim-batch-size', default=50, type=int,
                        help='batch size for model decision evaluation')
    parser.add_argument('--max-query', default=10000, type=int,
                        help='maximum number of queries allowed')
    parser.add_argument('--init-num-eval', default=25, type=int,
                        help='initial number of evaluations for gradient estimation')
    parser.add_argument('--max-num-eval', default=10000, type=int,
                        help='maximum number of generating scored samples for reinforcement learning')
    parser.add_argument('--fix-num-eval', action='store_true',
                        help='fix number of evaluations for gradient estimation to args.init_num_eval')
    parser.add_argument('--gamma', default=50, type=float,
                        help='to decide binary search threshold') # used
    parser.add_argument('--delta-mult', default=1.0, type=float,
                        help='to increase sampling radius delta')
    parser.add_argument('--epsilon', default='1.0', type=float, help='the lp perturbation bound')
    parser.add_argument('--epsilon-schema', default='fixed', type=str, choices=['fixed', 'sqrt', 'cosine'],
                        help='init_epsilon schema')
    parser.add_argument('--fixed-epsilon-mult', default=0.4, type=float,
                        help='use to determine init_epsilon when fixed epsilon schema is used')
    parser.add_argument('--cosine-epsilon-round', default=4, type=int,
                        help='repeat high to low cosine epsilon procedure args.cosine_epsilon_round times')
    parser.add_argument('--calibrate-clip', action='store_true',
                        help='set to True to calibrate the grad vector to alleviate the clipping effect')
    parser.add_argument('--use-pytorch-rng', action='store_true',
                        help='we use numpy rng and then load into pytorch by default, since we found an unknown bug '
                             'in pytorch 1.0.0 rng, which leads to worse final distance. you can set this arg true '
                             'to use pytorch rng directly')
    parser.add_argument('--save-grad', action='store_true',
                        help='save grad after attacking step, if the adv image is significantly changed')
    parser.add_argument('--save-grad-dir', type=str,
                        default='{}/train_pytorch_model/policy_driven_attack/pretrained_data'.format(PROJECT_PATH),
                        help='direction to save .npy files')
    parser.add_argument('--save-grad-pct', default=0.0, type=float,
                        help='save grad if distance args.save_grad_pct% in this iteration')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed') # used
    parser.add_argument('--batch-size', default=1, type=int)  # used
    parser.add_argument('--target_type', type=str, default='increment',
                        choices=['random', "load_random", 'least_likely', "increment"])
    parser.add_argument('--all-archs', action="store_true")
    parser.add_argument('--json-config', type=str, default='{}/TangentAttack-main/configures/PDA.json'.format(PROJECT_PATH),
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--ssh', action='store_true',
                        help='whether or not we are executing command via ssh.'
                             'If set to True, we will not print anything to screen and only redirect them to log file') # used
    parser.add_argument('--load-random-class-image', action='store_true',
                        help='load a random image from the target class')  # npz {"0":, "1": ,"2": }
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.try_split = list(map(float, args.try_split))
    args.ce_lmbd = list(map(float, args.ce_lmbd))
    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset][args.norm_type]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
        args_dict = defaults
    if args.targeted:
        if args.dataset == "ImageNet":
            args.max_queries = 20000
    return args


def print_args():
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))


def get_random_dir_name(seed=None):
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choice(vocab) for _ in range(8))


if __name__ == '__main__':
    # before going to the main function, we do following things:
    # 1. setup output directory
    # 2. make global variables: args, model (on cpu), loaders and device

    # 1. setup output directory
    args = parse_args()
    print(1)
    # if args.num_part > 1, then this experiment is just a part and we should use the same token for all parts
    # to guarantee that, we use sha256sum of config in string format to generate unique token
    assert 0 <= args.part_id < args.num_part <= args.num_image
    args.exp_dir = osp.join(args.exp_dir, 'PDA-{}-{}-{}'.format(args.dataset, args.norm_type, args.targeted))
    os.makedirs(args.exp_dir, exist_ok=True)

    # set log file, and import glog after that (since we might change sys.stdout/stderr on set_log_file())
    if args.all_archs:
        set_log_file(osp.join(args.exp_dir, 'run.log'), file_only=args.ssh)
    else:
        set_log_file(osp.join(args.exp_dir, 'run_{}.log'.format(args.victim_arch)), file_only=args.ssh)
    import glog as log
    log.info('Host: {}, user: {}, CUDA_VISIBLE_DEVICES: {}, cwd: {}'.format(
        socket.gethostname(), getpass.getuser(), os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.getcwd()))
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info('Called with args:')
    print_args()

    # 2. make global variables

    # check device
    device = torch.device('cuda')
    # set random seed before init model
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.all_archs:
        archs = MODELS_TEST_STANDARD[args.dataset]
    else:
        assert args.victim_arch is not None
        archs = [args.victim_arch]
    for victim_arch in archs:
        model = StandardModel(args.dataset, victim_arch, no_grad=False, load_pretrained=True).eval().to(device)
        victim = VictimQuery(model)
        attacker = PDAttack(victim, args.epsilon, args.external_init_adv_image, args.use_pytorch_rng, args.ssh,
                            args.dataset, args.targeted, victim_arch, args.phase, args.norm_type, args.victim_batch_size,
                            0, 1.0, 1)
        # do the business
        attacker.attack_all_images(args)