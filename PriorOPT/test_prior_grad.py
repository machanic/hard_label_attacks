import torch
from torch import nn
import numpy as np

def g_function_newton(model, theta, bound, create_graph=True):
    lmdb = torch.tensor(0.5, requires_grad=True)  # 初始化lmdb为0.0，设置为可求导张量
    perturbed = lmdb * theta
    Lp_distance = model(perturbed)
    loss = Lp_distance - bound
    while torch.abs(loss) > 1e-4:
        grad_lmdb = torch.autograd.grad(loss, lmdb, create_graph=create_graph)[0]
        lmdb = lmdb - loss / grad_lmdb
        perturbed = lmdb * theta
        Lp_distance = model(perturbed)
        loss = Lp_distance - bound
    return lmdb

# 可以作为soft label attack使用
def g_function_newton_with_est_grad_lmdb(model, theta, bound, sigma=0.001):
    lmdb = torch.tensor(0.5, requires_grad=True)  # 初始化lmdb为0.0，设置为可求导张量
    perturbed = lmdb * theta
    Lp_distance = model(perturbed)
    loss = Lp_distance - bound
    while torch.abs(loss) > 1e-3:
        # grad_lmdb = torch.autograd.grad(loss, lmdb, create_graph=create_graph)[0]
        loss_2 = model((lmdb + sigma) * theta) - bound
        est_grad_lmdb = (loss_2 - loss) / sigma
        lmdb = lmdb - loss / est_grad_lmdb
        perturbed = lmdb * theta
        Lp_distance = model(perturbed)
        loss = Lp_distance - bound
    return lmdb

def get_g_grad(model, theta, bound):
    with torch.enable_grad():
        theta.requires_grad_()
        min_lmdb = g_function_newton(model, theta, bound, True)
        grad_theta = torch.autograd.grad(min_lmdb, theta, retain_graph=False)[0]
    return grad_theta

def normalize(grad):
    return grad / torch.sqrt(torch.sum(torch.square(grad)))

def sign_opt_grad(model, theta, rv, target_bound=1.0, sigma=0.001):
    predict_labels = []
    u_batch = []
    initial_lmdb = g_function_newton(model, theta, target_bound, False).item()
    for i in range(rv.size(0)):
        u = rv[i]
        # u /= torch.linalg.norm(u, dim=-1, keepdim=True)
        new_theta = normalize(theta + sigma * u)
        # new_theta /= torch.linalg.norm(new_theta, dim=-1, keepdim=True)
        new_lmdb = g_function_newton(model, new_theta, target_bound, False).item()
        if new_lmdb > initial_lmdb:  # 按照原始的半径走，这个新的方向的分类边界更远，所以落在了分类器内部true_label处
            predict_labels.append(1)
        else:
            predict_labels.append(-1)  # 这个新方向的分类边界更近，所以按照原来的半径已经走出分类决策线了
        u_batch.append(u)

    u_batch = torch.stack(u_batch, 0)
    predict_labels = torch.tensor(predict_labels, dtype=torch.float32)
    predict_labels = predict_labels.view(u_batch.size(0),1)
    sign_grad = torch.mean(u_batch * predict_labels, dim=0)
    # sign_grad = sign_grad / rv.size(0)
    return sign_grad

def opt_grad(model, theta, rv, target_bound=1.0, sigma=0.001):
    loss = g_function_newton(model, theta, target_bound, create_graph=False)
    # new_thetas = theta.view(1,-1) + rv
    grad = torch.zeros(rv.size(1))
    for i in range(rv.size(0)):
        loss_new = g_function_newton(model, normalize(theta + sigma *rv[i]), target_bound, create_graph=False)
        grad += ((loss_new.item() - loss.item()) / sigma) * rv[i]
    return grad / rv.size(0)

def prior_opt_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
    prior_grad = get_g_grad(surrogate_model, theta, surrogate_bound)
    prior_grad = normalize(prior_grad)
    # true_grad = normalize(get_g_grad(model, theta, target_bound))
    loss = g_function_newton(model, theta, target_bound, create_graph=False)
    query = 1
    prior_new_theta = normalize(theta + sigma * prior_grad)
    prior_deriv = (g_function_newton(model, prior_new_theta, target_bound, create_graph=False) - loss) / sigma
    query += 1

    if prior_deriv < 0:
        prior_deriv, prior_grad = -prior_deriv, -prior_grad
    us = []
    q = rv.size(0)
    for i in range(q):
        us.append(rv[i])
    us.append(prior_grad)
    # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
    orthos = []
    for u in us:
        for ou in orthos:
            u = u - torch.sum(u * ou) * ou
        u = u / torch.sqrt(torch.sum(u * u))
        orthos.append(u)
    xs = torch.stack([normalize(theta + sigma * u) for u in orthos], dim=0)
    pert_losses = []
    for x in xs:
        pert_losses.append(g_function_newton(model, x, target_bound, create_graph=False))
        query += 1
    derivatives = [(pl - loss) / sigma for pl in pert_losses]
    grad_est = sum([deriv * u for deriv, u in zip(derivatives, orthos)])
    grad_est_ord_norm2 = sum([deriv ** 2 for deriv in derivatives[:q]])
    return grad_est

def prior_opt_grad_with_est_lmdb_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
    prior_grad = get_g_grad(surrogate_model, theta, surrogate_bound)
    prior_grad = normalize(prior_grad)
    # true_grad = normalize(get_g_grad(model, theta, target_bound))
    loss = g_function_newton_with_est_grad_lmdb(model, theta, target_bound, sigma=sigma)
    query = 1
    prior_new_theta = normalize(theta + sigma * prior_grad)
    prior_deriv = (g_function_newton_with_est_grad_lmdb(model, prior_new_theta, target_bound, sigma=sigma) - loss) / sigma
    query += 1

    if prior_deriv < 0:
        prior_deriv, prior_grad = -prior_deriv, -prior_grad
    us = []
    q = rv.size(0)
    for i in range(q):
        us.append(rv[i])
    us.append(prior_grad)
    # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
    orthos = []
    for u in us:
        for ou in orthos:
            u = u - torch.sum(u * ou) * ou
        u = u / torch.sqrt(torch.sum(u * u))
        orthos.append(u)
    xs = torch.stack([normalize(theta + sigma * u) for u in orthos], dim=0)
    pert_losses = []
    for x in xs:
        pert_losses.append(g_function_newton_with_est_grad_lmdb(model, x, target_bound, sigma=sigma))
        query += 1
    derivatives = [(pl - loss) / sigma for pl in pert_losses]
    grad_est = sum([deriv * u for deriv, u in zip(derivatives, orthos)])
    grad_est_ord_norm2 = sum([deriv ** 2 for deriv in derivatives[:q]])
    return grad_est


def prior_opt_sign_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
    prior_grad = get_g_grad(surrogate_model, theta, surrogate_bound)
    prior_grad = normalize(prior_grad)
    # true_grad = normalize(get_g_grad(model, theta, target_bound))
    loss = g_function_newton(model, theta, target_bound, create_graph=False)
    query = 1
    prior_new_theta = normalize(theta + sigma * prior_grad)
    prior_deriv = (g_function_newton(model, prior_new_theta, target_bound, create_graph=False) - loss) /sigma
    query += 1

    if prior_deriv < 0:
        prior_deriv, prior_grad = -prior_deriv, -prior_grad
    us = []
    q = rv.size(0)
    for i in range(q):
        us.append(rv[i])
    us.append(prior_grad)
    # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
    orthos = []
    for u in us:
        for ou in orthos:
            u = u - torch.sum(u * ou) * ou
        u = u / torch.sqrt(torch.sum(u * u))
        orthos.append(u)
    orthos = torch.stack(orthos, dim=0)

    grad_est = sign_opt_grad(model, theta, orthos, target_bound, sigma)
    return grad_est



def prior_opt_bs_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
    prior = get_g_grad(surrogate_model, theta, surrogate_bound)
    prior = prior / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior, prior))), min=1e-12)
    l = target_bound
    # compute norm_square
    s = 10
    pert = torch.randn(size=(s, rv.size(1)))
    for i in range(s):
        pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.sum(torch.mul(pert[i], pert[i]))), min=1e-12)
    new_thetas = (theta.view(1,-1) + sigma * pert) / torch.linalg.norm(theta.view(1,-1) + sigma * pert, dim=1,ord=2, keepdim=True)
    losses = torch.zeros(s)
    for i in range(s):
        new_theta = new_thetas[i]
        g_i = g_function_newton(model, new_theta, target_bound, create_graph=False)
        losses[i] = g_i.detach()
    norm_square = torch.mean(torch.square((losses - l) / sigma))  # scalar
    while True:
        prior_new_theta = (theta + sigma * prior) / torch.linalg.norm(theta + sigma * prior)
        prior_loss = g_function_newton(model, prior_new_theta, target_bound, create_graph=False)
        diff_prior = (prior_loss - l)[0].item()
        if diff_prior == 0:
            sigma *= 2
            print("sigma={:.4f}, multiply sigma by 2".format(sigma))
        else:
            break
    est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior, prior)) * norm_square),
                                                 min=1e-12)
    est_alpha = est_alpha.item()
    alpha = est_alpha  # alpha描述了替代模型的梯度是否有用，alpha越大λ也越大，λ=1表示相信这个prior
    if alpha < 0:  # 夹角大于90度，cos变成负数
        prior = -prior  # v = -v , negative the transfer gradient,
        alpha = -alpha
    n = theta.size(1)
    K = rv.size(0)
    if alpha ** 2 <= 1 / (n + 2 * K - 2):
        lmda = 0
    elif alpha ** 2 >= (2 * K - 1) / (n + 2 * K - 2):
        lmda = 1
    else:
        lmda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * K - 2)) / (
                alpha ** 4 * n * (n + 2 * K - 2) - 2 * alpha ** 2 * n * K + 1)
    return_prior = False
    if lmda == 1:
        return_prior = True  # lmda =1, we trust this prior as true gradient
    if not return_prior:
        pert = rv
        for i in range(K):
            pert[i] = pert[i] - torch.sum(pert[i] * prior) / torch.clamp(torch.sum(prior * prior), min=1e-12) * prior
            pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.sum(torch.mul(pert[i], pert[i]))), min=1e-12)
            pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior  # paper's Algorithm 1: line 9
        while True:
            sign_grad = sign_opt_grad(model, theta, pert, target_bound, sigma)
            norm_grad = torch.sqrt(torch.sum(torch.mul(sign_grad, sign_grad)))
            if norm_grad.item() == 0:
                sigma *= 5
                print("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
            else:
                break
        return sign_grad
    else:
        return prior


class HyperBallModel(nn.Module):

    def __init__(self, ball_type="l2"):
        super().__init__()
        self.ball_type = ball_type


    def generate_symmetric_positive_definite_matrix(self, dim):
        A = torch.randn(dim, dim)
        A = torch.matmul(A, A.t())
        return A

    def is_symmetric_positive_definite_matrix(self, matrix):
        # 检查矩阵是否对称
        if not torch.allclose(matrix, matrix.t()):
            print("不对称")
            return False
        # 计算矩阵的特征值
        eigenvalues, _ = torch.eig(matrix)
        # 检查特征值是否都大于零
        if not torch.all(eigenvalues[:, 0] > 0):
            print("不是正定矩阵")
            return False

        return True

    def check_point_in_ellipsoid(self, point, axis_lengths):
        # 计算对角矩阵
        diag_matrix = torch.diag(1.0 / (axis_lengths ** 2))

        # 生成一个n x n的随机正交矩阵，用于旋转椭球体
        rotation_matrix = torch.randn(point.size(0), point.size(0))
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix)

        # 将点应用到椭球体的坐标系中
        point_transformed = torch.matmul(rotation_matrix, point)

        # 判断点是否在椭球体内部
        is_inside = torch.matmul(torch.matmul(point_transformed, diag_matrix), point_transformed) <= 1.0

        return is_inside

    def forward(self, x):
        if self.ball_type == "l1":
            return torch.linalg.norm(x, dim=-1, ord=1)
        elif self.ball_type == "l2":
            return torch.linalg.norm(x,dim=-1, ord=2)
        elif self.ball_type == "l0.5":
            return torch.linalg.norm(x, dim=-1, ord=0.5)
        elif self.ball_type == "linf":
            return torch.linalg.norm(x, dim=-1,ord=float('inf'))



if __name__ == "__main__":

    dim = 100
    q = 50
    sigma = 0.001
    theta = torch.randn(dim)
    theta = normalize(theta)
    cos_opt = 0
    cos_sign_opt = 0
    cos_prior_opt = 0
    cos_prior_sign_opt = 0
    cos_prior_opt_with_est_lmdb_grad = 0

    for_loop = 100
    target_bound = 10
    surrogate_bound = 10
    model = HyperBallModel("l0.5")
    surrogate_model_l1 = HyperBallModel("l1")
    surrogate_model_linf = HyperBallModel("linf")
    surrogate_model_l2 = HyperBallModel("l2")


    true_grad = normalize(get_g_grad(model, theta,target_bound))
    for iii in range(for_loop):
        print(iii)
        rv = torch.randn(q, dim)
        rv = rv / torch.norm(rv, dim=1, keepdim=True)
        est_grad_opt = normalize(opt_grad(model, theta, rv, target_bound=target_bound, sigma=sigma))
        cos_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_opt, dim=-1).item()
        est_grad_sign_opt = normalize(sign_opt_grad(model, theta, rv, target_bound=target_bound, sigma=sigma))
        cos_sign_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_sign_opt, dim=-1).item()

        est_grad_prior_opt = normalize(prior_opt_grad(model, surrogate_model_l1, theta, rv, surrogate_bound=surrogate_bound,
                                                      target_bound=target_bound, sigma=sigma))
        cos_prior_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_opt, dim=-1).item()
        est_grad_prior_sign_opt = normalize(prior_opt_sign_grad(model, surrogate_model_l1, theta, rv, surrogate_bound=surrogate_bound,
                                                                target_bound=target_bound, sigma=sigma))
        cos_prior_sign_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_sign_opt, dim=-1).item()

        est_grad_prior_opt_with_est_lmdb_grad = normalize(
            prior_opt_grad_with_est_lmdb_grad(model, surrogate_model_l1, theta, rv, surrogate_bound=surrogate_bound,
                           target_bound=target_bound, sigma=sigma))
        cos_prior_opt_with_est_lmdb_grad += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_opt_with_est_lmdb_grad, dim=-1).item()

    print("cos between true and opt: {}".format(cos_opt/ for_loop))
    print("cos between true and sign-opt: {}".format(cos_sign_opt/ for_loop))
    print("cos between true and prior-opt: {}".format(cos_prior_opt/for_loop))
    print("cos between true and prior-sign-opt: {}".format(cos_prior_sign_opt / for_loop))
    print("cos between true and prior-opt_with_est_lmdb_grad: {}".format(cos_prior_opt_with_est_lmdb_grad / for_loop))


