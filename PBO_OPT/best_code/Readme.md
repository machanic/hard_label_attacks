

## L2范数优化

#### GP设置
不需要噪声项

#### 标签规范化
`-((-y).log())`  ->  `(y - y.mean()) / y.std()`


#### 采样函数
`LatinHypercube`和`torch.quasirandom.SobolEngine`不如高斯随机采样

Adam 效果一般， Note: 使用时，改为-loss

    lr = 0.005
    beta = 3
    n_iter_opt_acq = 10

    for i in range(n_iter_opt_acq):
        objective = torch.mean(gen_objective(x))
        objective.backward()

        x.data = torch.clamp(x + lr * x.grad.sign(), min=self.lower, max=self.upper)
        x.grad.zero_()
    x = x / self.norm(x)

UCB和EI效果差不多


#### BayesOpt超参数
不建议设置`n_past`