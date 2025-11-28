import gpytorch
import torch
import numpy as np
import glog as log


def corr(a: torch.Tensor, b: torch.Tensor):
    return torch.mean(standardize(a) * standardize(b)).item()

def standardize(y: torch.Tensor):
    y = -1/y
    return (y - y.mean()) / y.std()


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, device, exact_fval, lengthscale=None, lengthscale_ini=None, prior_f=None, scale='adapt'):
        if exact_fval:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(0, 1e-6))
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
                self.register_parameter(name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1)))
                self.register_constraint('raw_scale', gpytorch.constraints.Positive())
            elif scale.startswith('fixed'):
                self.scale = 1.
            else:
                raise ValueError
        self.prior_f = prior_f
        self.jitter = 1e-4
        lengthscale = gpytorch.constraints.Interval(lengthscale, lengthscale + 0.0001) if lengthscale else None
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_constraint=lengthscale))
        if lengthscale_ini:
            self.covar_module.base_kernel.lengthscale = lengthscale_ini
        self.scale_type = scale

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        covar = covar.add_jitter(self.jitter)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

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

    def fit_hyper(self, history_y, prior_history_y, setto1=False, positivescale=True, training_iter=50):

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
        self.call_count = torch.zeros([1])

    def __call__(self, tensor, ldb):
        y, q = self.func(tensor, ldb)#.detach()
        if self.prior_func:
            prior_y = self.prior_func(tensor,0.5,use_parallel_search=True).detach()
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
        self.prior_f = prior_f
        self.scale = scale
        self.positivescale = positivescale

    def initialize_model(self, x_init=None, y_init=None):
        if self.f.func.is_targeted:
            model = GP(x_init, y_init, self.device, self.exact_fval, self.lengthscale, lengthscale_ini=0.01, prior_f=self.prior_f, scale=self.scale)
        else:
            model = GP(x_init, y_init, self.device, self.exact_fval, self.lengthscale, lengthscale_ini=np.sqrt(self.dim), prior_f=self.prior_f, scale=self.scale)
        if self.prior_f:
            self.set_prior_normalizer()
        return model

    def initialize(self, x_init=None):
        if x_init is not None:
            y_init = self.f(x_init, ldb=30)
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
            self.std =(-1/self.f.prior_history_y).std().item()
        else:
            self.mean = 0.
            self.std = 1.

    def normalize_prior_f(self, inp):
        inp = -1 / inp
        return (inp - self.mean) / self.std * self.model.scale + self.model.constant

    def gen_rand_init(self, n_rand_init=10, eps=1e-12):
        x = torch.empty(n_rand_init, self.dim).uniform_(-1.0, 1.0)
        norms = x.abs().amax(dim=1, keepdim=True).clamp_min(eps)
        return x / norms

    def gen_past_best(self, n_past_best=10):
        return self.f.history_x[-n_past_best:, :] if self.f.call_count > 0 and n_past_best > 0 else None

    def update_model(self, x):
        x = x.unsqueeze(0) if x.ndimension() == 1 else x
        y = self.f(x, 30)
        self.refresh_model()

    def refresh_model(self):
        y_obs = standardize(self.f.history_y) if self.normalize_y else self.f.history_y
        y_prior = self.normalize_prior_f(self.f.prior_history_y) if self.prior_f else 0.
        self.model.set_train_data(self.f.history_x, y_obs - y_prior, strict=False)

    def select_beta(self, iter, success_iter):
        delta = 0.05 + 0.1 * (
                    1 - sum(success_iter) / len(success_iter))  # δ越小，算法越保守（探索更多），但收敛更稳健；δ越大，算法越激进（利用更多），但可能错过全局最优。
        return np.sqrt(2 * np.log((iter + 1) ** 2 * np.pi ** 2 / (6 * delta)))

    def find_next(self, iter, success_iter):
        lr = 0.1                                            # fixme
        beta = self.select_beta(iter, success_iter[-20:])
        n_iter_opt_acq = 30

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
                pred = self.model(x / torch.norm(x, dim=tuple(range(1, x.ndim)), p=float('inf'), keepdim=True))  # FIXME
                ans = pred.mean + beta * torch.sqrt(pred.variance)
            else:
                ans = 0.
            if self.prior_f:
                opt = self.f.get_opt()
                res = self.prior_f(x, ldb=-opt, use_parallel_search=True)
                ans += self.normalize_prior_f(res)
            return ans
        # 优化过程
        for i in range(n_iter_opt_acq):
            objective = torch.mean(gen_objective(x))  # 计算目标函数
            objective.backward()
            x.data = torch.clamp(x + lr * x.grad.sign(), min=-1, max=1)
            x.grad.zero_()
        # 返回优化后的候选点
        objective = gen_objective(x)
        return x[torch.argsort(objective, descending=True)[:self.n_opt], :]

    def fit_hyper(self, setto1=False):
        if self.prior_f:
            self.set_prior_normalizer()
            log.info('Correlation between target and prior: {}'.format(corr(self.f.prior_history_y, self.f.history_y)))
        loss_now = self.model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None, setto1=setto1, positivescale=self.positivescale)
        ini_model = self.initialize_model(self.model.train_inputs, self.model.train_targets)
        # whether to setto1 in ini_model.fit_hyper? answer: no need
        loss_ini = ini_model.fit_hyper(standardize(self.f.history_y), standardize(self.f.prior_history_y) if self.prior_f else None, setto1=setto1, positivescale=self.positivescale)
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
                    # print('{:.2f}MB, {:.2f}MB'.format(torch.cuda.memory_allocated('cuda:0') / 1048576, torch.cuda.memory_reserved('cuda:0') / 1048576))
                    if i % freq_fit_hyper == 0 and self.scale != 'fixed_only':
                        setto1 = (i == 0) if self.positivescale else False
                        self.fit_hyper(setto1=setto1)
                    x = self.find_next(i, success_iter).detach()
                    self.update_model(x)
                    opt = self.f.get_opt()
                    newest = self.f.get_newest()
                    msg = f'{self.f.func.idx+1}-th Image, iter={i} q={self.f.call_count.item()} best_f={-opt:.4f} f={-newest:.4f}'
                    if self.prior_f:
                        msg += f" f'={-self.f.get_prior_newest()}"
                    log.info(msg)
                    opts.append(opt)
                    newests.append(newest)
                    if newest == opt:
                        success_iter.append(1)
                        save_dist(abs(opt), self.f.call_count)
                    else:
                        success_iter.append(0)
                    if self.f.call_count >= maximum_queries:
                        x = self.f.get_opt_x() * abs(opt)
                        return x, opts, newests
                return None, opts, newests

