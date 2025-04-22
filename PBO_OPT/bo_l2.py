import gpytorch
import torch
import numpy as np
import argparse
import os
import glog as log
import math
from torch.distributions import Normal

def corr(a: torch.Tensor, b: torch.Tensor):
    return torch.mean(standardize(a) * standardize(b)).item()

def standardize(y: torch.Tensor):
    # y = -((-y).log())
    # y = y.mean() / y
    y = -1/y
    return (y - y.mean()) / y.std()
    # lambd=-2.
    # y=-((-y)**lambd-1)/lambd
    # return y#(y - y.mean()) / y.std()


class TruncatedGaussianLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    def __init__(self, lower_bound=None, upper_bound=None, noise_constraint=None):
        super(TruncatedGaussianLikelihood, self).__init__(noise_constraint=noise_constraint)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, function_values, targets, **kwargs):
        """
        计算截断高斯分布的对数似然（带上界和下界）。
        """
        normal_dist = Normal(function_values, torch.ones_like(function_values))

        # 计算原始的对数似然
        log_likelihood = normal_dist.log_prob(targets)

        # 计算下界截断部分：目标值大于下界时保留
        if self.lower_bound is not None:
            truncation_term_lower = normal_dist.log_prob(torch.tensor(self.lower_bound).to(function_values.device))
            log_likelihood -= truncation_term_lower

        # 计算上界截断部分：目标值小于上界时保留
        if self.upper_bound is not None:
            truncation_term_upper = normal_dist.log_prob(torch.tensor(self.upper_bound).to(function_values.device))
            log_likelihood -= truncation_term_upper

        return log_likelihood.sum()


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, device, exact_fval, lengthscale=None, lengthscale_ini=None, prior_f=None, scale='adapt'):
        if exact_fval:
            likelihood = TruncatedGaussianLikelihood(upper_bound=0, noise_constraint=gpytorch.constraints.Interval(0, 1e-6))
            # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, 1e-4))
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, 1e-6))
        if device == 'cuda':
            likelihood = likelihood.cuda()
        super().__init__(train_x, train_y, likelihood)
        if prior_f is None:
            self.mean_module = gpytorch.means.ConstantMean()
            self.constant = self.mean_module.constant
        else:
            self.mean_module = gpytorch.means.ZeroMean()
            self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(1)))
            if scale == 'adapt':
                self.register_parameter(name="scale", parameter=torch.nn.Parameter(torch.ones(1)))
            elif scale.startswith('fixed'):
                self.scale = 1.
            else:
                raise ValueError
        self.prior_f = prior_f
        lengthscale = gpytorch.constraints.Interval(lengthscale, lengthscale + 0.0001) if lengthscale else None
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=lengthscale))
        if lengthscale_ini:
            self.covar_module.base_kernel.lengthscale = lengthscale_ini
        self.scale_type = scale

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


    def fit_hyper(self, history_y, prior_history_y, setto1=False, positivescale=True):
        training_iter = 50

        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self(*[i.detach() for i in self.train_inputs])
            prior_targets = (self.scale * prior_history_y.detach() + self.constant) if self.prior_f else 0.
            loss = -mll(output, history_y.detach() - prior_targets)
            loss.backward()
            msg = 'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.3f   constant: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.covar_module.outputscale.item(),
                self.likelihood.noise.item(),
                self.constant.item()
            )
            if self.prior_f is not None:
                msg += '   scale: %.3f' % (self.scale if isinstance(self.scale, float) else self.scale.item())
            # if i % 10 == 9:
            #     print(msg)
            optimizer.step()
            if self.prior_f is not None and self.scale_type == 'adapt' and positivescale:
                if setto1:
                    self.scale.data = torch.ones(1)
                else:
                    self.scale.data = torch.clamp(self.scale, min=0.)

        return loss.item()


# class GP(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, device, exact_fval, lengthscale=None, lengthscale_ini=None, prior_f=None, scale='adapt'):
#         if exact_fval:
#             likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, 1e-6))
#         else:
#             likelihood = gpytorch.likelihoods.GaussianLikelihood()
#         if device == 'cuda':
#             likelihood = likelihood.cuda()
#         super().__init__(train_x, train_y, likelihood)
#         if prior_f is None:
#             self.mean_module = gpytorch.means.ConstantMean()
#             self.constant = self.mean_module.constant
#         else:
#             self.mean_module = gpytorch.means.ZeroMean()
#             self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(1)))
#             if scale == 'adapt':
#                 self.register_parameter(name="scale", parameter=torch.nn.Parameter(torch.ones(1)))
#             elif scale.startswith('fixed'):
#                 self.scale = 1.
#             else:
#                 raise ValueError
#         self.prior_f = prior_f
#         lengthscale = gpytorch.constraints.Interval(lengthscale, lengthscale + 0.0001) if lengthscale else None
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=lengthscale))
#         if lengthscale_ini:
#             self.covar_module.base_kernel.lengthscale = lengthscale_ini
#         self.scale_type = scale
#
#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
#
#
#     def fit_hyper(self, history_y, prior_history_y, setto1=False, positivescale=True):
#         training_iter = 50
#
#         self.train()
#         self.likelihood.train()
#
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
#
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
#
#         for i in range(training_iter):
#             optimizer.zero_grad()
#             output = self(*[i.detach() for i in self.train_inputs])
#             prior_targets = (self.scale * prior_history_y.detach() + self.constant) if self.prior_f else 0.
#             loss = -mll(output, history_y.detach() - prior_targets)
#             loss.backward()
#             msg = 'Iter %d/%d - Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.3f   constant: %.3f' % (
#                 i + 1, training_iter, loss.item(),
#                 self.covar_module.base_kernel.lengthscale.item(),
#                 self.covar_module.outputscale.item(),
#                 self.likelihood.noise.item(),
#                 self.constant.item()
#             )
#             if self.prior_f is not None:
#                 msg += '   scale: %.3f' % (self.scale if isinstance(self.scale, float) else self.scale.item())
#             # if i % 10 == 9:
#             #     log.info(msg)
#             optimizer.step()
#             if self.prior_f is not None and self.scale_type == 'adapt' and positivescale:
#                 if setto1:
#                     self.scale.data = torch.ones(1)
#                 else:
#                     self.scale.data = torch.clamp(self.scale, min=0.)
#
#         return loss.item()


class Func:
    def __init__(self, f, prior_f=None):
        self.func = f
        self.prior_func = prior_f
        self.call_count = torch.zeros([1])

    def __call__(self, tensor, ldb, save_query=True):
        y, q = self.func(tensor, ldb)#.detach()
        if save_query:
            if self.prior_func:
                prior_y = self.prior_func(tensor, self.func.epsilon).detach()
            if self.call_count.item() == 0:
                self.call_count += self.func.init_q
                self.history_x = tensor
                self.history_y = y
                if self.prior_func:
                    self.prior_history_y = prior_y
            else:
                self.history_x = torch.cat([self.history_x, tensor], dim=0)
                self.history_y = torch.cat([self.history_y, y], dim=0)
                if self.prior_func:
                    self.prior_history_y = torch.cat([self.prior_history_y, prior_y], dim=0)

            self.call_count += tensor.shape[0] + q
        return y

    def get_opt_x(self):
        return self.history_x[torch.argmax(self.history_y), :]

    def get_opt(self):
        return torch.max(self.history_y).item()

    def get_newest(self):
        return self.history_y[-1].item()

    def get_prior_newest(self):
        return self.prior_history_y[-1].item()

class MyOPT:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]  # 二阶矩
        self.t = 0  # 时间步

    def norm(self, t):
        assert len(t.shape) == 2
        norm_vec = torch.sqrt(t.pow(2).sum(dim=1)).view(-1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def sgd_step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad.data
            _x = torch.clamp(param.data + self.lr * g / self.norm(g), min=-1., max=1.)
            param.data = _x / self.norm(_x)

    def adam_step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g  # 更新一阶矩
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2  # 更新二阶矩
            m_hat = self.m[i] / (1 - self.beta1**self.t)  # 一阶矩偏差校正
            v_hat = self.v[i] / (1 - self.beta2**self.t)  # 二阶矩偏差校正
            _g = m_hat / (torch.sqrt(v_hat) + self.eps)  # 更新参数
            _x = torch.clamp(param.data + self.lr * _g, min=-1., max=1.)
            param.data = _x / self.norm(_x)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


class BayesOpt:
    def __init__(self, f, bounds, device='cuda', n_opt=1, n_init=10, n_past=10,
                 normalize_y=True, exact_fval=True, lengthscale=None, prior_f=None,
                 scale='adapt', positivescale=True):
        self.f = Func(f, prior_f=prior_f)
        self.bounds = bounds
        self.dim = len(self.bounds)
        self.device = device
        self.n_opt = n_opt
        self.n_init = n_init
        self.n_past = n_past
        self.normalize_y = normalize_y
        self.exact_fval = exact_fval
        self.lengthscale = lengthscale
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.lower = torch.tensor([bound[0] for bound in self.bounds]).unsqueeze(0)
        self.upper = torch.tensor([bound[1] for bound in self.bounds]).unsqueeze(0)
        self.prior_f = prior_f
        self.scale = scale
        self.positivescale = positivescale

    def initialize_model(self, x_init=None, y_init=None):
        model = GP(x_init, y_init, self.device, self.exact_fval, self.lengthscale, lengthscale_ini=None, prior_f=self.prior_f, scale=self.scale)
        # model = GP(x_init, y_init, self.device, self.exact_fval, self.lengthscale, lengthscale_ini=np.sqrt(self.dim), prior_f=self.prior_f, scale=self.scale)
        # model = GP(x_init, y_init, self.device, self.exact_fval, 0.693, lengthscale_ini=None, prior_f=self.prior_f, scale=self.scale)
        if self.prior_f:
            self.set_prior_normalizer()
        return model

    def initialize(self, x_init=None):
        if x_init is not None:
            # y_init = self.f(x_init, ldb=30.)
            y_init = self.f(x_init, ldb=0.1*math.sqrt(math.prod(self.f.func.data.shape)))
            if self.normalize_y:
                y_init = standardize(y_init)
        else:
            y_init = None
        self.model = self.initialize_model(x_init, y_init)
        if self.device == 'cuda':
            self.model = self.model.cuda()

    def set_prior_normalizer(self):
        if self.scale != 'fixed_only':
            self.mean = self.f.prior_history_y.mean().item()
            self.std = self.f.prior_history_y.std().item()
        else:
            self.mean = 0.
            self.std = 1.

    def normalize_prior_f(self, inp):
        return (inp - self.mean) / self.std * self.model.scale + self.model.constant

    def norm(self, t):
        assert len(t.shape) == 2
        norm_vec = torch.sqrt(t.pow(2).sum(dim=1)).view(-1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def gen_rand_init(self, n_rand_init=10):
        x_init = torch.rand([n_rand_init, self.dim])
        x_init = x_init * (self.upper - self.lower) + self.lower
        x_init = x_init / self.norm(x_init)
        return x_init

    def gen_past_best(self, n_past_best=10):
        return self.f.history_x[-n_past_best:, :] if self.f.call_count > 0 and n_past_best > 0 else None
        # _, indices = torch.topk(self.f.history_y, n_past_best)
        # return self.f.history_x[indices, :] if self.f.call_count > 0 and n_past_best > 0 else None

    def update_model(self, x):
        x = x.unsqueeze(0) if x.ndimension() == 1 else x
        y = self.f(x, abs(self.f.get_opt()))
        self.refresh_model()

    def refresh_model(self):
        y_obs = standardize(self.f.history_y) if self.normalize_y else self.f.history_y
        y_prior = self.normalize_prior_f(self.f.prior_history_y) if self.prior_f else 0.
        self.model.set_train_data(self.f.history_x, y_obs - y_prior, strict=False)

    def update_bounds(self, is_update=True):
        if self.f.func.is_targeted and is_update:
            best = self.f.get_opt_x().squeeze(0)
            self.upper = torch.clamp(best + 0.1, min=-1., max=1.)
            self.lower = torch.clamp(best - 0.1, min=-1., max=1.)
        else:
            pass

    def select_beta(self, iter):
        # return 3 * (1.001 ** iter)
        # return 3 * (1.1 ** (iter//100))
        return 3 * (1.25 ** (iter//200))

    def grad_uncertainty(self, X, x, model, likelihood, obs_noise=0.0001):
        """
        使用高斯过程模型计算多个查询点的不确定性（方差）。

        参数：
        X (tensor): 已知查询点的输入数据，形状为 (N, D)
        x (tensor): 当前查询点的输入数据，形状为 (M, D)
        model (gpytorch.models.ExactGP): 高斯过程回归模型
        likelihood (gpytorch.likelihoods.GaussianLikelihood): 高斯过程的观测噪声
        obs_noise (float): 观测噪声（默认 0.01）

        返回：
        tensor: 当前多个查询点 x 处的不确定性（方差），形状为 (M,)
        """
        x_tensor = x.clone().detach()  # 当前查询点 x，形状为 (M, D)

        # 设置为评估模式，不计算梯度
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # 预测当前多个查询点 x 的均值和协方差
            pred = model(x_tensor)  # 预测多个查询点的均值和协方差

            # 获取当前查询点的协方差矩阵（K_xx）
            K_xx = pred.covariance_matrix + obs_noise * torch.eye(len(x_tensor))  # 加入噪声项

            # 获取训练数据点 X 的协方差矩阵（K_XX）
            K_Xx = model.covar_module(x_tensor, X)  # X 与 x 的协方差，形状为 (M, N)
            K_XX = model.covar_module(X, X)  # X 与 X 的协方差，形状为 (N, N)

            # 将 LazyEvaluatedKernelTensor 转换为普通张量
            K_XX_eval = K_XX.evaluate()  # 计算出 K_XX 的实际值
            K_Xx_eval = K_Xx.evaluate()  # 计算出 K_Xx 的实际值

            # 计算 K_XX 的逆
            K_XX_inv = torch.linalg.inv(K_XX_eval + obs_noise * torch.eye(X.shape[0]))  # 加入噪声项，避免不可逆

            # 计算不确定性（方差），对于每个查询点分别计算
            var = K_xx - K_Xx_eval @ K_XX_inv @ K_Xx_eval.T  # 方差计算，形状为 (M, M)

            # 返回每个查询点的不确定性（方差），形状为 (M,)
            return var.diagonal().cpu()  # 返回对角线上的方差值（每个查询点的不确定性）

    def find_next(self, best, iter):
        # lr = 0.1
        # beta = 3        #self.select_beta(iter)
        # n_iter_opt_acq = 20
        #
        # self.model.eval()
        # self.model.likelihood.eval()
        #
        # x_rand_init = self.gen_rand_init(self.n_init)
        # x_past_best = self.gen_past_best(self.n_past)
        # x = torch.concat([x_rand_init, x_past_best]).clone().detach() if x_past_best is not None else x_rand_init
        # x.requires_grad_()
        #
        # def gen_objective(x):
        #     if not self.prior_f or self.scale != 'fixed_only':
        #         pred = self.model(x)
        #         ans = pred.mean + beta * torch.sqrt(torch.diag(pred.covariance_matrix))
        #     else:
        #         ans = 0.
        #     if self.prior_f:
        #         ans += self.normalize_prior_f(self.prior_f(x, self.f.func.epsilon))
        #     return ans
        #
        # for i in range(n_iter_opt_acq):
        #     objective = torch.mean(gen_objective(x))
        #     objective.backward()
        #
        #     # ZoRD
        #     # grads_uncertainty = self.grad_uncertainty(self.f.history_x, x, self.model, self.model.likelihood).detach()
        #     # idx = torch.logical_and(x_iter < 20, torch.logical_or(grads_uncertainty < 0.4,
        #     #                         torch.full_like(grads_uncertainty, i < 10, dtype=torch.bool))).detach()
        #     # x_iter[idx] += 1
        #     # print(i, idx, grads_uncertainty, x_iter)
        #
        #     _x = torch.clamp(x + lr * x.grad / self.norm(x.grad), min=self.lower, max=self.upper)
        #     x.data = _x / self.norm(_x)
        #     x.grad.zero_()
        #
        # #     x.data = torch.clamp(x + lr * x.grad.sign(), min=self.lower, max=self.upper)
        # #     x.data = x + lr * x.grad.sign()
        # #     x.grad.zero_()
        # # x = x / self.norm(x)
        #
        # objective = gen_objective(x)
        # # obj = self.f(x, save_query=False)
        # return x[torch.argsort(objective, descending=True)[:self.n_opt], :]
        lr = 0.1
        beta = self.select_beta(iter)
        # print(beta)
        n_iter_opt_acq = 20

        # 设置模型为评估模式
        self.model.eval()
        self.model.likelihood.eval()

        # 初始化候选点
        x_rand_init = self.gen_rand_init(self.n_init)
        x_past_best = self.gen_past_best(self.n_past)
        x = torch.concat([x_rand_init, x_past_best]).clone().detach() if x_past_best is not None else x_rand_init
        x.requires_grad_()

        # 定义目标函数
        def gen_objective(x):
            if not self.prior_f or self.scale != 'fixed_only':
                pred = self.model(x)
                ans = pred.mean + beta * torch.sqrt(torch.diag(pred.covariance_matrix))
            else:
                ans = 0.
            if self.prior_f:
                ans += self.normalize_prior_f(self.prior_f(x, self.f.func.epsilon))
            return ans

        # 使用 Adam 优化器
        optimizer = MyOPT([x], lr=lr)

        # 优化过程
        for i in range(n_iter_opt_acq):
            optimizer.zero_grad()  # 清零梯度
            objective = torch.mean(gen_objective(x))  # 计算目标函数
            objective.backward()  # 反向传播计算梯度
            optimizer.sgd_step()  # 更新参数

        # 返回优化后的候选点
        objective = gen_objective(x)
        return x[torch.argsort(objective, descending=True)[:self.n_opt], :]


    def fit_hyper(self, setto1=False):
        if self.prior_f:
            self.set_prior_normalizer()
            log.info('Correlation between target and prior: {}'.format(corr(self.f.prior_history_y, self.f.history_y)))
        loss_now = self.model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None, setto1=setto1, positivescale=self.positivescale)
        # print('------------------------------')
        ini_model = self.initialize_model(self.model.train_inputs, self.model.train_targets)
        # whether to setto1 in ini_model.fit_hyper? answer: no need
        loss_ini = ini_model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None, setto1=setto1, positivescale=self.positivescale)
        # log.info(f'loss now: {loss_now}, loss ini: {loss_ini}')
        if loss_ini < loss_now or np.isnan(loss_now):
            log.info('Loss reinitializing is lower; changing model')
            self.model = ini_model
        self.refresh_model()

    def run(self, n_iter=10000, freq_fit_hyper=10, save_dist=None):
        if self.scale != 'fixed_only':
            opts, newests = [self.f.get_opt()], [self.f.get_newest()]
        else:
            opts, newests = [], []
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            with gpytorch.settings.cholesky_max_tries(6):
                for i in range(n_iter):
                    # print('{:.2f}MB, {:.2f}MB'.format(torch.cuda.memory_allocated('cuda:0') / 1048576, torch.cuda.memory_reserved('cuda:0') / 1048576))
                    if i % freq_fit_hyper == 0 and self.scale != 'fixed_only':
                        setto1 = (i == 0) if self.positivescale else False
                        self.fit_hyper(setto1=setto1)
                    x = self.find_next(self.f.get_opt(), i).detach()
                    self.update_model(x)
                    opt = self.f.get_opt()
                    newest = self.f.get_newest()
                    self.update_bounds()
                    msg = f'{self.f.func.idx+1}-th Image, iter={i} q={self.f.call_count.item()} best_f={-opt:.4f} f={-newest:.4f}'
                    if self.prior_f:
                        msg += f" f'={self.f.get_prior_newest()}"
                    log.info(msg)
                    opts.append(opt)
                    newests.append(newest)
                    if newest == opt:
                        save_dist(abs(opt), self.f.call_count)
                    if self.f.call_count > n_iter:
                        x = self.f.get_opt_x() * abs(opt)
                        return x, opts, newests
                return None, opts, newests



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--method', type=str, choices=['no', 'fix', 'adapt'], default='no',
                        help='no: BO without prior_f, fix: P-BO with c=1, adapt: P-BO adapting c')
    parser.add_argument('--prior_func', type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help='choose the prior_f')
    parser.add_argument('--name', type=str,
                        help='name of the experiment (used in the filename of the saved npy)')

    dim = 768
    def target_func(tensor: torch.Tensor, iter):      # 1000 iters -> -1.1
        # return -torch.sum(torch.square(tensor), axis=-1)+1000*((torch.sum(torch.square(tensor), axis=-1)>700).float()-1)
        return -torch.sum(torch.square(tensor), axis=-1)+1000*math.pow(0.8, iter//100)*((torch.sum(torch.square(tensor), axis=-1)>700).float()-1)

    prior_bias = ((torch.arange(dim) + 1.) / dim).cuda()
    def prior_func1(tensor: torch.Tensor):      # 1000 iters -> -1.1
        return -torch.sum(torch.square(tensor - prior_bias), axis=-1)+1000*((torch.sum(torch.square(tensor-prior_bias), axis=-1)>30).float()-1)

    def prior_func2(tensor: torch.Tensor):
        return torch.sum(tensor, axis=-1)

    def prior_func3(tensor: torch.Tensor):
        return -torch.sum((torch.arange(tensor.shape[-1]) + 1.) * torch.square(tensor - prior_bias), axis=-1)

    def prior_func4(tensor: torch.Tensor):      # 330 iters -> -1
        return -target_func(tensor)

    def prior_func5(tensor: torch.Tensor):
        return -torch.sum(torch.square(tensor - 1.), axis=-1)

    args = parser.parse_args()
    print(args)
    bounds = [[-1, 1]] * dim
    if args.prior_func == 1:
        prior_f = prior_func1
    elif args.prior_func == 2:
        prior_f = prior_func2
    elif args.prior_func == 3:
        prior_f = prior_func3
    elif args.prior_func == 4:
        prior_f = prior_func4
    elif args.prior_func == 5:
        prior_f = prior_func5

    if args.method == 'no':
        prior_f = None
    scale = args.method
    if scale == 'fix':
        scale = 'fixed'
    bo = BayesOpt(target_func, bounds, device='cuda', n_opt=1, n_init=10, n_past=10,
                normalize_y=True, exact_fval=True, prior_f=prior_f, scale=scale, positivescale=False)
    bo.initialize(bo.gen_rand_init())
    _, opts, newests = bo.run(n_iter=2000)
    # os.makedirs('results', exist_ok=True)
    # np.save(f'results/{args.method}_{args.prior_func}_{args.name}_opt.npy', np.array(opts))
    # np.save(f'results/{args.method}_{args.prior_func}_{args.name}_newest.npy', np.array(newests))
