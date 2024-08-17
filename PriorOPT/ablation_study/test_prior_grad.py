import torch
from torch import nn
import numpy as np
import re
from sklearn.decomposition import PCA, KernelPCA


# 牛顿迭代法确定image沿θ方向到边界的距离，也可以用二分查找法
def g_function_newton(model, image, theta, create_graph=True):
    theta = normalize(theta)
    with torch.enable_grad():
        lmdb = torch.tensor(0.5, requires_grad=True)  # initialize lmdb as 0.5
        perturbed = image + lmdb * theta
        loss = model(perturbed)[:,1]
        while torch.abs(loss) > 1e-3:
            grad_lmdb = torch.autograd.grad(loss, lmdb, create_graph=create_graph)[0] # note that it should use True for high-order gradient
            lmdb = lmdb - loss / grad_lmdb
            perturbed = image + lmdb * theta
            loss = model(perturbed)[:,1]
    return lmdb
#

def cw_loss(logit, label, target=None):
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

def get_g_grad(model, image, theta, use_lambda_grad=False):
    with torch.enable_grad():
        min_lmdb = g_function_newton(model, image, theta, create_graph=use_lambda_grad)
        if use_lambda_grad:
            return torch.autograd.grad(min_lmdb, theta,create_graph=False)[0]
        theta = theta.detach()
        theta.requires_grad_()
        loss = cw_loss(model(image + min_lmdb * normalize(theta)), 0, None)
        grad_theta = torch.autograd.grad(-loss, theta, create_graph=False)[0]
    return grad_theta

def normalize(grad):
    return grad / torch.sqrt(torch.sum(torch.square(grad)))

def sign_opt_grad(model, image, theta, rv,  sigma=0.001):
    predict_labels = []
    u_batch = []
    initial_lmdb = g_function_newton(model, image, theta, create_graph=False).item()
    for i in range(rv.size(0)):
        u = rv[i]
        # u /= torch.linalg.norm(u, dim=-1, keepdim=True)
        new_theta = normalize(theta + sigma * u)
        # new_theta /= torch.linalg.norm(new_theta, dim=-1, keepdim=True)
        new_lmdb = g_function_newton(model, image, new_theta, create_graph=False).item()
        if new_lmdb > initial_lmdb:  # 按照原始的半径走，这个新的方向的分类边界更远，所以落在了分类器内部true_label处
            predict_labels.append(1)
        else:
            predict_labels.append(-1)  # 这个新方向的分类边界更近，所以按照原来的半径已经走出分类决策线了
        u_batch.append(u)
    u_batch = torch.stack(u_batch, 0)
    predict_labels = torch.tensor(predict_labels, dtype=torch.float32)
    predict_labels = predict_labels.view(u_batch.size(0),1)
    sign_grad = torch.mean(u_batch * predict_labels, dim=0)
    return sign_grad

def opt_grad(model, image, theta, rv,  sigma=0.001):
    loss = g_function_newton(model, image, theta,  create_graph=False)
    # new_thetas = theta.view(1,-1) + rv
    grad = torch.zeros(rv.size(1))
    for i in range(rv.size(0)):
        loss_new = g_function_newton(model, image, normalize(theta + sigma * rv[i]), create_graph=False)
        grad += ((loss_new.item() - loss.item()) / sigma) * rv[i]
    return grad / rv.size(0)


def prior_sign_opt_grad(model, surrogate_models, image, theta, rv, sigma=0.001):
    us = []
    for surrogate_model in surrogate_models:
        prior_grad = get_g_grad(surrogate_model, image, theta, use_lambda_grad=False)
        prior_grad = normalize(prior_grad)
        us.append(prior_grad.squeeze())
    q = rv.size(0) - len(surrogate_models)
    for i in range(q):
        us.append(rv[i])
    # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
    orthos = []
    for u in us:
        for ou in orthos:
            u = u - torch.sum(u * ou) * ou
        u = u / torch.sqrt(torch.sum(u * u))
        orthos.append(u)
    orthos = torch.stack(orthos, dim=0)

    predict_labels = []
    u_batch = []
    initial_lmdb = g_function_newton(model, image, theta, create_graph=False).item()
    for i in range(orthos.size(0)):
        u = orthos[i]
        # u /= torch.linalg.norm(u, dim=-1, keepdim=True)
        new_theta = normalize(theta + sigma * u)
        # new_theta /= torch.linalg.norm(new_theta, dim=-1, keepdim=True)
        new_lmdb = g_function_newton(model, image, new_theta, create_graph=False).item()
        if new_lmdb > initial_lmdb:  # 按照原始的半径走，这个新的方向的分类边界更远，所以落在了分类器内部true_label处
            predict_labels.append(1)
        else:
            predict_labels.append(-1)  # 这个新方向的分类边界更近，所以按照原来的半径已经走出分类决策线了
        u_batch.append(u)
    u_batch = torch.stack(u_batch, 0)
    predict_labels = torch.tensor(predict_labels, dtype=torch.float32)
    predict_labels = predict_labels.view(u_batch.size(0), 1)
    sign_grad = torch.mean(u_batch * predict_labels, dim=0)
    return sign_grad

def prior_opt_grad(model, surrogate_models, image, theta, rv,  sigma=0.001):
    prior_grads = []
    for surrogate_model in surrogate_models:
        prior_grad = get_g_grad(surrogate_model, image, theta, use_lambda_grad=False)
        prior_grad = normalize(prior_grad)
        prior_grads.append(prior_grad.squeeze())
    us = []
    q = rv.size(0) - len(surrogate_models)
    for prior_grad in prior_grads:
        us.append(prior_grad)
    for i in range(q):
        us.append(rv[i])
    # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
    orthos = []
    for u in us:
        for ou in orthos:
            u = u - torch.sum(u * ou) * ou
        u = u / torch.sqrt(torch.sum(u * u))
        orthos.append(u)
    u_batch = []
    initial_lmdb = g_function_newton(model, image, theta, create_graph=False).item()
    predict_labels = []
    # finite_difference_dim = len(prior_grads)
    finite_difference_dim = 2

    for orth in orthos[finite_difference_dim:]:
        new_theta = theta + sigma * orth
        new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
        new_lmdb = g_function_newton(model, image, new_theta, create_graph=False).item()
        if new_lmdb > initial_lmdb:  # 按照原始的半径走，这个新的方向的分类边界更远，所以落在了分类器内部true_label处
            predict_labels.append(1)
        else:
            predict_labels.append(-1)  # 这个新方向的分类边界更近，所以按照原来的半径已经走出分类决策线了
        u_batch.append(orth)
    u_batch = torch.stack(u_batch, 0)
    predict_labels = torch.tensor(predict_labels, dtype=torch.float32)
    predict_labels = predict_labels.view(u_batch.size(0), 1)
    sign_grad = torch.mean(u_batch * predict_labels, dim=0,keepdim=True)
    sign_grad /= torch.sqrt(torch.sum(sign_grad * sign_grad))
    # perform score-based RGF gradient estimation, and we need to perform Gram-Schmidt orthogonalization again.
    # sign_grad_ortho = sign_grad - sum(torch.sum(sign_grad * ou) * ou for ou in orthos[0:finite_difference_dim])
    # sign_grad_ortho = sign_grad_ortho / torch.sqrt(torch.sum(sign_grad_ortho * sign_grad_ortho))
    new_orthos = orthos[0:finite_difference_dim] + [sign_grad]
    est_grad = torch.zeros_like(theta)
    for grad_theta_orth in new_orthos:
        new_theta = theta + sigma * grad_theta_orth
        new_theta = normalize(new_theta)
        perturb_bound = g_function_newton(model, image, new_theta, create_graph=False)
        weight = (perturb_bound - initial_lmdb) / sigma
        est_grad += weight * grad_theta_orth
    est_grad = est_grad / len(new_orthos)
    return est_grad

# def prior_pca_ensemble_opt_grad(model, surrogate_models, image, theta, rv,  sigma=0.001):
#     prior_grads = []
#     for surrogate_model in surrogate_models:
#         prior_grad = get_g_grad(surrogate_model, image, theta, use_lambda_grad=True)
#         prior_grad = normalize(prior_grad)
#         prior_grads.append(prior_grad.squeeze())
#     us = []
#     q = rv.size(0)
#     for prior_grad in prior_grads:
#         us.append(prior_grad)
#     for i in range(q - len(surrogate_models)):
#         us.append(rv[i])
#     X = torch.stack(us).cpu().detach()
#     pca = PCA(n_components=q)
#     pca.fit(X)
#     orthos = torch.from_numpy(pca.components_).float()  # shape = (q,d)
#     finite_difference_dim = 2
#     u_batch = []
#     predict_labels = []
#     initial_lmdb = g_function_newton(model, image, theta, create_graph=False).item()
#     for orth in orthos[finite_difference_dim:]:
#         u = orth.view_as(theta)
#         new_theta = theta + sigma * u
#         new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
#         new_lmdb = g_function_newton(model, image, new_theta, create_graph=False).item()
#         if new_lmdb > initial_lmdb:  # 按照原始的半径走，这个新的方向的分类边界更远，所以落在了分类器内部true_label处
#             predict_labels.append(1)
#         else:
#             predict_labels.append(-1)  # 这个新方向的分类边界更近，所以按照原来的半径已经走出分类决策线了
#         u_batch.append(orth)
#     u_batch = torch.stack(u_batch, 0)
#     predict_labels = torch.tensor(predict_labels, dtype=torch.float32)
#     predict_labels = predict_labels.view(u_batch.size(0), 1)
#     sign_grad = torch.mean(u_batch * predict_labels, dim=0, keepdim=True)
#     # perform score-based RGF gradient estimation, and we need to perform Gram-Schmidt orthogonalization again.
#     sign_grad_ortho = sign_grad - sum(torch.sum(sign_grad * ou) / torch.sum(ou * ou) * ou for ou in orthos[0:finite_difference_dim])
#     sign_grad_ortho = sign_grad_ortho / torch.sqrt(torch.sum(sign_grad_ortho * sign_grad_ortho))
#     new_orthos = []
#     for orth in orthos[0:finite_difference_dim]:
#         new_orthos.append(orth.view_as(theta))
#     new_orthos.append(sign_grad_ortho.view_as(theta))
#     est_grad = torch.zeros_like(theta)
#     for grad_theta_orth in new_orthos:
#         new_theta = theta + sigma * grad_theta_orth
#         new_theta = normalize(new_theta)
#         perturb_bound = g_function_newton(model, image, new_theta, create_graph=False)
#         weight = (perturb_bound - initial_lmdb) / sigma
#         est_grad += weight * grad_theta_orth
#     return est_grad

# def prior_opt_grad_with_est_lmdb_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
#     prior_grad = get_g_grad(surrogate_model, theta, surrogate_bound)
#     prior_grad = normalize(prior_grad)
#     # true_grad = normalize(get_g_grad(model, theta, target_bound))
#     loss = g_function_newton_with_est_grad_lmdb(model, theta, target_bound, sigma=sigma)
#     query = 1
#     prior_new_theta = normalize(theta + sigma * prior_grad)
#     prior_deriv = (g_function_newton_with_est_grad_lmdb(model, prior_new_theta, target_bound, sigma=sigma) - loss) / sigma
#     query += 1
#
#     if prior_deriv < 0:
#         prior_deriv, prior_grad = -prior_deriv, -prior_grad
#     us = []
#     q = rv.size(0)
#     for i in range(q):
#         us.append(rv[i])
#     us.append(prior_grad)
#     # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
#     orthos = []
#     for u in us:
#         for ou in orthos:
#             u = u - torch.sum(u * ou) * ou
#         u = u / torch.sqrt(torch.sum(u * u))
#         orthos.append(u)
#     xs = torch.stack([normalize(theta + sigma * u) for u in orthos], dim=0)
#     pert_losses = []
#     for x in xs:
#         pert_losses.append(g_function_newton_with_est_grad_lmdb(model, x, target_bound, sigma=sigma))
#         query += 1
#     derivatives = [(pl - loss) / sigma for pl in pert_losses]
#     grad_est = sum([deriv * u for deriv, u in zip(derivatives, orthos)])
#     grad_est_ord_norm2 = sum([deriv ** 2 for deriv in derivatives[:q]])
#     return grad_est



# def prior_opt_bs_grad(model, surrogate_model, theta, rv, surrogate_bound=1.0, target_bound=1.0, sigma=0.001):
#     prior = get_g_grad(surrogate_model, theta, surrogate_bound)
#     prior = prior / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior, prior))), min=1e-12)
#     l = target_bound
#     # compute norm_square
#     s = 10
#     pert = torch.randn(size=(s, rv.size(1)))
#     for i in range(s):
#         pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.sum(torch.mul(pert[i], pert[i]))), min=1e-12)
#     new_thetas = (theta.view(1,-1) + sigma * pert) / torch.linalg.norm(theta.view(1,-1) + sigma * pert, dim=1,ord=2, keepdim=True)
#     losses = torch.zeros(s)
#     for i in range(s):
#         new_theta = new_thetas[i]
#         g_i = g_function_newton(model, new_theta, target_bound, create_graph=False)
#         losses[i] = g_i.detach()
#     norm_square = torch.mean(torch.square((losses - l) / sigma))  # scalar
#     while True:
#         prior_new_theta = (theta + sigma * prior) / torch.linalg.norm(theta + sigma * prior)
#         prior_loss = g_function_newton(model, prior_new_theta, target_bound, create_graph=False)
#         diff_prior = (prior_loss - l)[0].item()
#         if diff_prior == 0:
#             sigma *= 2
#             print("sigma={:.4f}, multiply sigma by 2".format(sigma))
#         else:
#             break
#     est_alpha = diff_prior / sigma / torch.clamp(torch.sqrt(torch.sum(torch.mul(prior, prior)) * norm_square),
#                                                  min=1e-12)
#     est_alpha = est_alpha.item()
#     alpha = est_alpha  # alpha描述了替代模型的梯度是否有用，alpha越大λ也越大，λ=1表示相信这个prior
#     if alpha < 0:  # 夹角大于90度，cos变成负数
#         prior = -prior  # v = -v , negative the transfer gradient,
#         alpha = -alpha
#     n = theta.size(1)
#     K = rv.size(0)
#     if alpha ** 2 <= 1 / (n + 2 * K - 2):
#         lmda = 0
#     elif alpha ** 2 >= (2 * K - 1) / (n + 2 * K - 2):
#         lmda = 1
#     else:
#         lmda = (1 - alpha ** 2) * (1 - alpha ** 2 * (n + 2 * K - 2)) / (
#                 alpha ** 4 * n * (n + 2 * K - 2) - 2 * alpha ** 2 * n * K + 1)
#     return_prior = False
#     if lmda == 1:
#         return_prior = True  # lmda =1, we trust this prior as true gradient
#     if not return_prior:
#         pert = rv
#         for i in range(K):
#             pert[i] = pert[i] - torch.sum(pert[i] * prior) / torch.clamp(torch.sum(prior * prior), min=1e-12) * prior
#             pert[i] = pert[i] / torch.clamp(torch.sqrt(torch.sum(torch.mul(pert[i], pert[i]))), min=1e-12)
#             pert[i] = np.sqrt(1 - lmda) * pert[i] + np.sqrt(lmda) * prior  # paper's Algorithm 1: line 9
#         while True:
#             sign_grad = sign_opt_grad(model, theta, pert, target_bound, sigma)
#             norm_grad = torch.sqrt(torch.sum(torch.mul(sign_grad, sign_grad)))
#             if norm_grad.item() == 0:
#                 sigma *= 5
#                 print("estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(sigma))
#             else:
#                 break
#         return sign_grad
#     else:
#         return prior


class HyperBallModel(nn.Module):

    def __init__(self, ball_type="l2", class_bound=10):
        super().__init__()
        self.ball_type = ball_type
        self.class_bound = class_bound

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
            distance = torch.linalg.norm(x, dim=-1, ord=1)
        elif self.ball_type == "l2":
            distance = torch.linalg.norm(x,dim=-1, ord=2)
        elif "l0." in self.ball_type:
            norm = float(self.ball_type[1:])
            distance = torch.linalg.norm(x, dim=-1, ord=norm)
        elif self.ball_type == "linf":
            distance = torch.linalg.norm(x, dim=-1,ord=float('inf'))  # B,d
        logits = torch.stack([self.class_bound - distance, distance - self.class_bound],dim=-1)  # B,2d
        return logits


if __name__ == "__main__":

    dim = 100
    q = 50
    sigma = 0.001
    image = torch.zeros(1,dim, requires_grad=True)
    cos_opt = 0
    cos_sign_opt = 0
    cos_prior_opt = 0
    cos_prior_opt_single = 0
    cos_prior_sign_opt = 0
    #
    for_loop = 100
    target_bound = torch.tensor(10.0)
    surrogate_bound = torch.tensor(10.0)
    target_model = HyperBallModel("l1", target_bound)
    surrogate_model_l1 = HyperBallModel("l1", surrogate_bound)
    surrogate_model_l9 = HyperBallModel("l0.9", surrogate_bound)
    surrogate_model_l7 = HyperBallModel("l0.7", surrogate_bound)
    surrogate_model_l6 = HyperBallModel("l0.6", surrogate_bound)
    surrogate_model_l5 = HyperBallModel("l0.5", surrogate_bound)
    surrogate_model_l4 = HyperBallModel("l0.4", surrogate_bound)
    surrogate_model_l3 = HyperBallModel("l0.3", surrogate_bound)
    surrogate_model_l02 = HyperBallModel("l0.2", surrogate_bound)
    surrogate_model_l01 = HyperBallModel("l0.1", surrogate_bound)
    surrogate_model_linf = HyperBallModel("linf", surrogate_bound)
    surrogate_model_l2 = HyperBallModel("l2", surrogate_bound)
    # surrogate_models = [surrogate_model_l4,  surrogate_model_l5,surrogate_model_l6,surrogate_model_l9,surrogate_model_l7] #, surrogate_model_l3, surrogate_model_l02,surrogate_model_l01, surrogate_model_l1]
    surrogate_models = [surrogate_model_l6, surrogate_model_l4]
    for iii in range(for_loop):
        print(iii)
        theta = torch.randn(1, dim, requires_grad=True)
        theta = normalize(theta)
        true_grad = normalize(get_g_grad(target_model, image, theta, use_lambda_grad=True))
        rv = torch.randn(q, dim)
        rv = rv / torch.norm(rv, dim=1, keepdim=True)
        est_grad_opt = normalize(opt_grad(target_model, image, theta, rv, sigma=sigma))
        cos_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_opt, dim=-1).item()
        est_grad_sign_opt = normalize(sign_opt_grad(target_model, image, theta, rv, sigma=sigma))
        cos_sign_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_sign_opt, dim=-1).item()

        est_grad_prior_opt = normalize(prior_opt_grad(target_model, surrogate_models, image, theta, rv, sigma=sigma))
        cos_prior_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_opt, dim=-1).item()

        # est_grad_prior_opt_single = normalize(prior_opt_grad(model_half_norm, [surrogate_model_l6], image, theta, rv, sigma=sigma))
        # cos_prior_opt_single += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_opt_single, dim=-1).item()

        est_grad_prior_sign_opt = normalize(prior_sign_opt_grad(target_model, surrogate_models, image, theta, rv, sigma=sigma))
        cos_prior_sign_opt += torch.nn.functional.cosine_similarity(true_grad, est_grad_prior_sign_opt, dim=-1).item()

    print("cos between true and opt: {}".format(cos_opt/ for_loop))
    print("cos between true and sign-opt: {}".format(cos_sign_opt/ for_loop))
    print("cos between true and prior-opt: {}".format(cos_prior_opt/for_loop))
    # print("cos between true and prior-single-opt: {}".format(cos_prior_opt_single / for_loop))
    print("cos between true and prior-sign-opt: {}".format(cos_prior_sign_opt / for_loop))
    # print("surrogate model set:{}".format(len(surrogate_models)))

