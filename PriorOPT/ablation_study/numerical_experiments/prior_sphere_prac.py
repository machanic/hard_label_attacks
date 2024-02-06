import os
import random

import numpy as np
import math
from scipy.special import gammaln
from tqdm import tqdm
import torch
import argparse

def compute_prod(prior, w, q, d, count=1000):
    vs = torch.cat((prior.unsqueeze(0).unsqueeze(-1).repeat(count, 1, 1), torch.randn(size=(count, d, q), device='cuda')), dim=-1)
    vs = torch.linalg.qr(vs, mode='reduced')[0][:, :, 1:]
    inner_prod = torch.matmul(vs.transpose(-2, -1), w) # b, q
    sumed = (torch.sign(inner_prod.unsqueeze(1)) * vs).sum(dim=-1) # b, d
    sumed = sumed / torch.linalg.norm(sumed, dim=-1, keepdim=True)
    grad_est = torch.matmul(sumed, w).unsqueeze(-1) * sumed + torch.matmul(prior, w) * prior
    grad_est = grad_est / torch.linalg.norm(grad_est, dim=-1, keepdim=True)
    angle = torch.matmul(grad_est, w).detach().cpu().numpy().tolist()
    return angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prior-OPT')
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--gpu', type=int, required=True, help='which GPU ID will be used')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ortho = True
    d = 3072
    q = 50
    w = torch.randn(size=(d,), device='cuda')
    w = w / torch.linalg.norm(w)
    alpha = 0.1
    rand = torch.randn(size=(d,), device='cuda')
    rand_perp = rand - torch.sum(rand * w) * w
    rand_perp = rand_perp / torch.linalg.norm(rand_perp)
    prior = alpha * w + np.sqrt(1 - alpha ** 2) * rand_perp
    assert np.abs(torch.linalg.norm(prior).item() - 1) < 1e-4
    assert np.abs(torch.sum(prior * w).item() - alpha) < 1e-4

    def single(d):
        return np.exp(gammaln(d / 2) - gammaln((d+1) / 2)) / np.sqrt(math.pi)

    mean_product = np.sqrt(alpha ** 2 + (single(d-1) * np.sqrt(1 - alpha ** 2) * np.sqrt(q)) ** 2)
    print('lower bound of theory mean', mean_product)
    mean_square = alpha ** 2 + 1 / (d - 1) * (1 + 2 / math.pi * (q - 1)) * (1 - alpha ** 2)
    print('theory square', mean_square)

    results = []
    for _ in tqdm(range(args.niter)):
        results += compute_prod(prior, w, q, d, count=1000)
    print(len(results))
    print('empirical mean', np.mean(results), '+-', np.std(results, ddof=1) / np.sqrt(len(results)))
    print('empirical mean square', np.mean(np.array(results) ** 2), '+-', np.std(np.array(results) ** 2, ddof=1) / np.sqrt(len(results)))
