import torch
import numpy as np

def normalize(grad):
    return grad / torch.sqrt(torch.sum(torch.square(grad)))


class PGZOO:
    def __init__(self, target_model, bounds, device='cuda', prior_model=None, q=1, lr=0.25, sigma=0.5, sampler=None):
        self.target = target_model
        self.bounds = bounds
        self.prior = prior_model
        self.dim = len(self.bounds)
        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.lower = torch.tensor([bound[0] for bound in self.bounds])
        self.upper = torch.tensor([bound[1] for bound in self.bounds])
        self.q = q
        self.sigma = sigma
        self.lr = lr
        self.sampler = sampler
        self.total_q = 0

    def initialize_x(self, n_rand_init=10):
        x = torch.rand([n_rand_init, self.dim]) * (self.upper - self.lower) + self.lower
        vals = self.target(x)
        self.total_q += n_rand_init
        return x[torch.argmax(vals), :]

    def run(self, ini_x, query_limit, verbose=True):
        x = ini_x.detach()
        i = 0
        while self.total_q <= query_limit:
            target_grad = self.target.get_grad(x)
            loss = self.target(x)
            self.total_q += 1
            print(i, loss)
            if loss.item() > 0:
                print('Total queries:', self.total_q)
                return x
            if self.prior is not None:
                prior_grad = self.prior.get_grad(x)
                if verbose:
                    print('Prior angle: {:.4f} {:.4f}'.format(torch.sum(normalize(prior_grad) * normalize(target_grad)).item(), np.sqrt(1 / x.nelement())))
                prior_deriv = (self.target(x + self.sigma * normalize(prior_grad)) - loss) / self.sigma
                self.total_q += 1
                if prior_deriv < 0:
                    prior_deriv, prior_grad = -prior_deriv, -prior_grad
                if i != 0 and prior_deriv ** 2 >= grad_est_ord_norm2 and i - est_ite <= 10:
                    print('Good enough prior, directly use it as grad est')
                    x.data = torch.clamp(x + self.lr * prior_grad.sign(), min=self.lower, max=self.upper)
                    i += 1
                    continue

            us = []
            for _ in range(self.q):
                if self.sampler is None:
                    rnd = torch.randn_like(x)
                else:
                    rnd = self.sampler()
                us.append(rnd)
            if self.prior is not None:
                us.append(prior_grad)

            # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
            orthos = []
            for u in us:
                for ou in orthos:
                    u = u - torch.sum(u * ou) * ou
                u = u / torch.sqrt(torch.sum(u * u))
                orthos.append(u)

            xs = torch.stack([x + self.sigma * u for u in orthos], dim=0)
            pert_losses = self.target(xs)
            self.total_q += len(orthos)
            derivatives = [(pl - loss) / self.sigma for pl in pert_losses]
            grad_est = sum([deriv * u for deriv, u in zip(derivatives, orthos)])
            grad_est_ord_norm2 = sum([deriv ** 2 for deriv in derivatives[:self.q]])
            est_ite = i
            if verbose:
                print('Final angle: {:.4f} {:.4f}'.format(torch.sum(normalize(grad_est) * normalize(target_grad)).item(), np.sqrt(self.q / x.nelement())))

            print('Current queries:', self.total_q)
            x.data = torch.clamp(x + self.lr * grad_est.sign(), min=self.lower, max=self.upper)
            i += 1

        return None












    def prior_grad(self, images, theta, initial_lbd, true_label, target_label=None, sigma=0.001):
        assert images.dim() == 4
        assert theta.dim() == 4
        query = 0
        prior_grads = []
        if self.surrogate_model is not None:
            prior_grad = self.get_g_grad(self.surrogate_model, images, theta, true_label, target_label)
            prior_grads.append(prior_grad)
        else:
            # 貌似这种做法一步梯度就走到最好的地方了
            for surrogate_model in self.surrogate_models:
                prior_grad = self.get_g_grad(surrogate_model, images, theta, true_label, target_label)
                prior_grads.append(prior_grad)

        prior_grad = prior_grad / torch.norm(prior_grad, p=2, dim=(1, 2, 3), keepdim=True)
        prior_new_theta = (theta + sigma * prior_grad) / self.norm(theta + sigma * prior_grad)
        pred_prior_label = self.model(images + initial_lbd * prior_new_theta).max(1)[1].item()
        query += 1
        prior_deriv = 1
        if target_label is None:
            if pred_prior_label != true_label:
                prior_deriv = -1
        else:
            if pred_prior_label == target_label:
                prior_deriv = -1
        if prior_deriv < 0:
            prior_deriv, prior_grad = -prior_deriv, -prior_grad
        us = []
        for i in range(self.k):
            rv = torch.randn_like(theta.squeeze())
            rv = rv / torch.norm(rv.view(-1), p=2, dim=0)
            us.append(rv)
        us.append(prior_grad.squeeze())
        # Gram-Schmidt. You can also call torch.linalg.qr to perform orthogonalization
        orthos = []
        for u in us:
            for ou in orthos:
                u = u - torch.sum(u * ou) * ou
            u = u / torch.sqrt(torch.sum(u * u))
            orthos.append(u)
        orthos = torch.stack(orthos, dim=0)

        images_batch = []
        u_batch = []
        for orth in orthos:
            u = orth.unsqueeze(0)
            new_theta = theta + sigma * u
            new_theta /= torch.norm(new_theta.view(-1), p=2, dim=0)
            u_batch.append(orth)
            images_batch.append(images + initial_lbd * new_theta)
            query += 1
        images_batch = torch.cat(images_batch, 0)
        u_batch = torch.stack(u_batch, 0)  # B,C,H,W
        assert u_batch.dim() == 4
        sign = torch.ones(orthos.size(0), device='cuda')
        if target_label is not None:
            target_labels = torch.tensor([target_label for _ in range(orthos.size(0))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels == target_labels] = -1
        else:
            true_labels = torch.tensor([true_label for _ in range(orthos.size(0))], device='cuda').long()
            predict_labels = self.model(images_batch).max(1)[1]
            sign[predict_labels != true_labels] = -1
        sign_grad = torch.sum(u_batch * sign.view(orthos.size(0), 1, 1, 1), dim=0, keepdim=True)

        sign_grad = sign_grad / orthos.size(0)

        return sign_grad, query
