import os
import random

import numpy as np
import math
from scipy.special import gammaln
from tqdm import tqdm
import torch
import argparse

def compute_prod(w, q, d, count=1000):
    vs = torch.randn(size=(count, d, q), device='cuda')
    vs = torch.linalg.qr(vs, mode='reduced')[0]
    inner_prod = torch.matmul(vs.transpose(-2, -1), w) # b, q
    sumed = (torch.sign(inner_prod.unsqueeze(1)) * vs).sum(dim=-1) # b, d
    sumed = sumed / torch.linalg.norm(sumed, dim=-1, keepdim=True)
    angle = torch.matmul(sumed, w).detach().cpu().numpy().tolist()
    return angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign-OPT baseline')
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

    mean_value_1 = np.sqrt(1 / math.pi) * np.exp(gammaln(d / 2) - gammaln((d+1) / 2))
    print('theory mean', mean_value_1 * np.sqrt(q))
    print('upper bound of theory mean square', 1 / d + (q - 1) * (mean_value_1 ** 2))
    print('theory mean square', 1 / d + (q - 1) * 2 / math.pi / d)

    results = []
    for _ in tqdm(range(args.niter)):
        results += compute_prod(w, q, d, count=1000)
    print(len(results))
    print('empirical mean', np.mean(results), '+-', np.std(results, ddof=1) / np.sqrt(len(results)))
    print('empirical mean square', np.mean(np.array(results) ** 2), '+-', np.std(np.array(results) ** 2, ddof=1) / np.sqrt(len(results)))
