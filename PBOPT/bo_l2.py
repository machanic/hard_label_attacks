import gpytorch
import numpy as np
import argparse
import glog as log
import math
from torch.distributions import Normal
import torch

def corr(a: torch.Tensor, b: torch.Tensor):
    return torch.mean(standardize(a) * standardize(b)).item()

def standardize(y: torch.Tensor):
    # y = y / y.mean().abs()
    # y = -((-y).log())
    y = -1/y
    return (y - y.mean()) / y.std()



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
            likelihood = TruncatedGaussianLikelihood(upper_bound=None, noise_constraint=gpytorch.constraints.Interval(0, 1e-6))
            # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, max(1e-6, train_y.std() * 0.01)))
            # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5, 1e-3))
        if device == 'cuda':
            likelihood = likelihood.cuda()
        super().__init__(train_x, train_y, likelihood)
        self.jitter = 1e-4
        if prior_f is None:
            self.mean_module = gpytorch.means.ConstantMean()
            self.constant = self.mean_module.constant
        else:
            self.mean_module = gpytorch.means.ZeroMean()
            self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(1)))
            if scale == 'adapt':
                self.register_parameter(name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1)))
                self.register_constraint('raw_scale', gpytorch.constraints.Positive())
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

    @property
    def scale(self):
        # 自动使用约束的 transform（内部是 softplus）
        if getattr(self, 'scale_type', None) == 'adapt':
            return self.raw_scale_constraint.transform(self.raw_scale)  # >0
        return torch.as_tensor(1, device=self.constant.device)

    def set_scale_init(self, value: float):
        with torch.no_grad():
            init = torch.tensor(value, device=self.raw_scale.device, dtype=self.raw_scale.dtype)
            self.raw_scale.copy_(self.raw_scale_constraint.inverse_transform(init))

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        covar = covar.add_jitter(self.jitter)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


    def fit_hyper(self, history_y, prior_history_y, setto1=False, positivescale=True,  training_iter=50):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        if self.prior_f is not None and self.scale_type == 'adapt' and positivescale and setto1:
            self.set_scale_init(1.0)

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
            optimizer.step()
        return loss.item()


class Func:
    def __init__(self, f, prior_f=None):
        self.func = f
        self.prior_func = prior_f
        self.call_count = torch.zeros([1],dtype=torch.int)

    def __call__(self, tensor, ldb):
        y, q = self.func(tensor, ldb)#.detach()
        if self.prior_func:
            # print( self.prior_func(tensor, ldb))
            prior_y = self.prior_func(tensor, ldb, use_parallel_search=True)
        if self.call_count.item() == 0:
            # self.call_count += self.func.init_q
            self.history_x = tensor
            self.history_y = y
            if self.prior_func:
                self.prior_history_y = prior_y
        else:
            self.history_x = torch.cat([self.history_x, tensor], dim=0)
            self.history_y = torch.cat([self.history_y, y], dim=0)
            if self.prior_func:
                self.prior_history_y = torch.cat([self.prior_history_y, prior_y], dim=0)

        self.call_count += q
        return y

    def get_opt_x(self):
        return self.history_x[torch.argmax(self.history_y), :]

    def get_opt(self):
        return torch.max(self.history_y).item()

    def get_newest(self):
        return self.history_y[-1].item()

    def get_prior_newest(self):
        return self.prior_history_y[-1].item()

class AFOPT(object):
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
            g = param.grad.data / self.norm(param.grad.data)
            _x = torch.clamp(param.data + self.lr * g, min=-1., max=1.)
            param.data = _x / self.norm(_x)

    def adam_step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad.data / self.norm(param.grad.data)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g  # 更新一阶矩
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)  # 更新二阶矩
            m_hat = self.m[i] / (1 - self.beta1**self.t)  # 一阶矩偏差校正
            v_hat = self.v[i] / (1 - self.beta2**self.t)  # 二阶矩偏差校正
            _g = m_hat / (torch.sqrt(v_hat) + self.eps)  # 更新参数
            _x = torch.clamp(param.data + self.lr * _g, min=-1., max=1.)
            param.data = _x / self.norm(_x)
            self.m[i] /= self.norm(self.m[i])
            self.v[i] /= self.norm(self.v[i])

    def geo_step(self):
        """
        X: 当前点 (单位向量), shape [n, d]
        grads: 目标函数在欧氏空间的梯度, shape [n, d]
        lr: 学习率
        返回: 更新后的点 (仍在球面上), shape [n, d]
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            # 获取当前参数和梯度
            x = param.data
            grad = param.grad.data

            # 投影梯度到切空间
            tangent_grad = grad - (x * grad).sum(dim=1, keepdim=True) * x

            # 计算切向量范数
            grad_norm = torch.norm(tangent_grad, dim=1, keepdim=True)
            mask = grad_norm < 1e-8  # 形状 [b, 1]
            mask = mask.expand_as(tangent_grad)  # 扩展为 [b, l]

            # 避免除以零
            tangent_grad[mask] = 0
            # 指数映射更新
            step_size = self.lr * grad_norm
            param.data = torch.cos(step_size) * x + torch.sin(step_size) * tangent_grad / grad_norm

    def _riemannian_grad(self, x, grad):
        """批量黎曼梯度投影 (形状 [B,D] -> [B,D])"""
        # x: [B,D], grad: [B,D]
        # 计算切空间投影: grad - x * <x, grad>
        inner_prod = torch.einsum('bd,bd->b', x, grad).unsqueeze(-1)  # [B,1]
        return grad - x * inner_prod  # [B,D]

    def _exp_map(self, x, v):
        """批量指数映射 (形状 [B,D] -> [B,D])"""
        # x: [B,D], v: [B,D] (切向量)
        norm_v = torch.norm(v, p=2, dim=-1, keepdim=True)  # [B,1]
        norm_v = torch.clamp(norm_v, min=1e-8)
        return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)

    def _projection(self, x):
        """批量投影到单位球面 (形状 [B,D] -> [B,D])"""
        return x / torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-8)

    def riemannian_adam_step(self):
        for param in self.params:
            if param.grad is None:
                continue

            # 当前参数和梯度 (形状 [B,D])
            x = param.data  # [B,D]
            grad = param.grad.data/self.norm(param.grad.data)  # [B,D]

            # 步骤1: 计算批量黎曼梯度
            r_grad = self._riemannian_grad(x, grad)  # [B,D]

            # 步骤2: 计算更新方向并应用指数映射
            update_direction = -self.lr * r_grad  # [B,D]
            new_x = self._exp_map(x, update_direction)  # [B,D]

            # 步骤3: 数值稳定性投影
            param.data = self._projection(new_x)  # [B,D]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()


class BayesOpt:
    def __init__(self, f, bounds, device='cuda', n_opt=1, n_init=10, n_past=10,
                 normalize_y=True, exact_fval=False, lengthscale=None, prior_f=None,
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
        self.prior_f = prior_f
        self.scale = scale
        self.positivescale = positivescale

    def initialize_model(self, x_init=None, y_init=None):
        model = GP(x_init, y_init, self.device, self.exact_fval, self.lengthscale, lengthscale_ini=np.sqrt(self.dim)*0.02,  # fixme
                   prior_f=self.prior_f, scale=self.scale)
        if self.prior_f:
            self.set_prior_normalizer()
        return model

    def initialize(self, x_init=None):
        if x_init is not None:
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
            self.mean = (-1/self.f.prior_history_y).mean().item()
            self.std = (-1/self.f.prior_history_y).std().item()
            # self.mean = self.f.prior_history_y.mean().item()
            # self.std =  self.f.prior_history_y.std().item()
        else:
            self.mean = 0.
            self.std = 1.

    def normalize_prior_f(self, inp):
        # inp = inp * self.model.scale + self.model.constant
        inp = -1/inp
        # return (inp - inp.mean()) / inp.std() * self.model.scale + self.model.constant
        # print((self.f.prior_history_y))
        # print(self.mean,self.std,inp.mean(),inp.std())
        # return (inp - self.mean) / self.std
        assert self.std >= 0
        return (inp - self.mean) / self.std * self.model.scale + self.model.constant      # FIXME, 当inp是0的反而超过了

    def gen_rand_init(self, n_rand_init=10, eps=1e-12):
        x = torch.empty(n_rand_init, self.dim).uniform_(-1.0, 1.0)
        norms = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        return x / norms

    def gen_past_best(self, n_past_best=10):
        return self.f.history_x[-n_past_best:, :] if self.f.call_count > 0 and n_past_best > 0 else None

    def update_model(self, x):
        with torch.no_grad():
            x = x.unsqueeze(0) if x.ndimension() == 1 else x
            y = self.f(x, abs(self.f.get_opt()))
            self.refresh_model()

    def refresh_model(self):
        y_obs = standardize(self.f.history_y) if self.normalize_y else self.f.history_y
        y_prior = self.normalize_prior_f(self.f.prior_history_y) if self.prior_f else 0.
        # print('y_obs', y_obs, 'y_prior', y_prior)
        # print(y_prior)
        self.model.set_train_data(self.f.history_x, y_obs - y_prior, strict=False)



    def select_beta(self, iter, success_iter):
        delta = 0.05 + 0.1 * (1 - sum(success_iter) / len(success_iter))  # δ越小，算法越保守（探索更多），但收敛更稳健；δ越大，算法越激进（利用更多），但可能错过全局最优。
        return np.sqrt(2 * np.log((iter+1)**2 * np.pi**2 / (6 * delta)))

    def find_next(self, iter, success_iter):
        lr = 0.1                                            # fixme
        beta = self.select_beta(iter, success_iter[-10:])
        # beta =  100
        n_iter_opt_acq = 10                                 # fixme

        # 设置模型为评估模式
        self.model.eval()
        self.model.likelihood.eval()

        # 初始化候选点
        x_rand_init = self.gen_rand_init(n_rand_init=self.n_init)
        x_past_best = self.gen_past_best(n_past_best=self.n_past)
        x = torch.concat([x_rand_init, x_past_best]).clone().detach() if x_past_best is not None else x_rand_init
        x.requires_grad_()
        # 定义目标函数
        def gen_objective(x):
            if not self.prior_f or self.scale != 'fixed_only':
                pred = self.model(x / torch.norm(x, dim=tuple(range(1, x.ndim)), keepdim=True))
                p = pred.mean
                ans = p + beta * torch.sqrt(pred.variance)
            else:
                ans = 0.
            if self.prior_f:
                opt = self.f.get_opt()
                res = self.prior_f(x, ldb=-opt, use_parallel_search=True)
                ans += self.normalize_prior_f(res)
            return ans

        # 使用优化器
        optimizer = AFOPT([x], lr=lr)

        # 优化过程
        for i in range(n_iter_opt_acq):
            objective = torch.mean(gen_objective(x))  # 计算目标函数
            objective.backward()  # 反向传播计算梯度
            optimizer.sgd_step()  # 更新参数
            optimizer.zero_grad()
        # 返回优化后的候选点
        objective = gen_objective(x)
        result = x[torch.argsort(objective, descending=True)[:self.n_opt], :]
        return result


    def fit_hyper(self, setto1=False):
        if self.prior_f:
            self.set_prior_normalizer()
            log.info('Correlation between target and prior: {}'.format(corr(self.f.prior_history_y, self.f.history_y)))
        loss_now = self.model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None,
                                        setto1=setto1, positivescale=self.positivescale)

        ini_model = self.initialize_model(self.model.train_inputs, self.model.train_targets)
        # whether to setto1 in ini_model.fit_hyper? answer: no need
        loss_ini = ini_model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None,
                                       setto1=setto1, positivescale=self.positivescale, training_iter=50)
        self.ini_model = ini_model
        if loss_ini < loss_now or np.isnan(loss_now):
            log.info('Loss reinitializing is lower; changing model')
            self.model = ini_model
        self.refresh_model()

    def run(self, maximum_queries=10000, freq_fit_hyper=10, save_dist=None):
        if self.scale != 'fixed_only':
            opts, newests = [self.f.get_opt()], [self.f.get_newest()]
        else:
            opts, newests = [], []

        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            with gpytorch.settings.cholesky_max_tries(6):
                success_iter = [1]
                for i in range(maximum_queries):
                    if (i % freq_fit_hyper == 0) and self.scale != 'fixed_only':
                        setto1 = (i == 0) if self.positivescale else False  # 如果positivescale,则setto1第一轮迭代是True，其他几轮是False
                        self.fit_hyper(setto1=setto1)
                    x = self.find_next(i, success_iter).detach()
                    self.update_model(x)
                    opt = self.f.get_opt()
                    newest = self.f.get_newest()
                    opts.append(opt)
                    newests.append(newest)
                    if newest == opt:
                        msg = f'{self.f.func.idx + 1}-th Image, iter={i} q={self.f.call_count.item()} best_f={-opt:.10f} f={-newest:.4f}'
                        if self.prior_f:
                            msg += f" f'={-self.f.get_prior_newest()}"
                        log.info(msg)
                        success_iter.append(1)
                        save_dist(abs(opt), self.f.call_count)
                    else:
                        success_iter.append(0)
                    del x
                    if self.f.call_count >= maximum_queries:
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
                normalize_y=True, exact_fval=True, prior_f=prior_f, scale=scale, positivescale=True)
    bo.initialize(bo.gen_rand_init())
    _, opts, newests = bo.run(maximum_queries=2000)
    # os.makedirs('results', exist_ok=True)
    # np.save(f'results/{args.method}_{args.prior_func}_{args.name}_opt.npy', np.array(opts))
    # np.save(f'results/{args.method}_{args.prior_func}_{args.name}_newest.npy', np.array(newests))
