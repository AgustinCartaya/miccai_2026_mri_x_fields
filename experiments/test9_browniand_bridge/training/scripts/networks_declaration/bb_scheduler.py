import torch
from tqdm import tqdm



class Scheduler:
    def __init__(self, num_timesteps=1000, sample_step=200, s=1., sample_type="linear",
                 objective_type="noise", eta=1., seed=0):
        self.num_timesteps = num_timesteps
        self.s = s
        self.sample_type = sample_type
        self.sample_step = sample_step
        self.objective_type = objective_type
        self.eta = eta
        self.seed = seed

        if self.sample_type == "linear":
            m_min, m_max = 0.001, 0.999
            self.m_t = torch.linspace(m_min, m_max, self.num_timesteps)
        else:
            raise NotImplementedError

        self.m_tminus = torch.cat((torch.tensor([0.]), self.m_t[:-1]))
        self.variance_t = 2. * (self.m_t - self.m_t ** 2) * self.s
        self.variance_tminus = torch.cat((torch.tensor([0.]), self.variance_t[:-1]))
        ratio = ((1. - self.m_t) / (1. - self.m_tminus)).pow(2)
        self.variance_t_tminus = self.variance_t - self.variance_tminus * ratio
        self.posterior_variance_t = self.variance_t_tminus * self.variance_tminus / self.variance_t

        if self.sample_type == 'linear':
            midsteps = torch.linspace(self.num_timesteps - 1, 2, self.sample_step - 2, dtype=torch.long)
            self.steps = torch.cat((midsteps, torch.tensor([1, 0])))

        self.gen_backward_diffusion = torch.Generator().manual_seed(seed)

    def _move_buffers_to(self, device):
        self.m_t = self.m_t.to(device)
        self.m_tminus = self.m_tminus.to(device)
        self.variance_t = self.variance_t.to(device)
        self.variance_tminus = self.variance_tminus.to(device)
        self.variance_t_tminus = self.variance_t_tminus.to(device)
        self.posterior_variance_t = self.posterior_variance_t.to(device)
        self.steps = self.steps.to(device)

    def forward_diffusion(self, t, x_0, x_T, noise):
        device = x_0.device
        self._move_buffers_to(device)

        m_t = self.m_t[t].view(-1, *([1] * (x_0.dim() - 1)))
        sigma_t = self.variance_t[t].sqrt().view(-1, *([1] * (x_0.dim() - 1)))

        if self.objective_type == "noise":
            objective = noise
        elif self.objective_type == "grad":
            objective = m_t * (x_T - x_0) + sigma_t * noise
        elif self.objective_type == "ysubx":
            objective = x_T - x_0

        x_next = (1. - m_t) * x_0 + m_t * x_T + sigma_t * noise
        return x_next, objective

    def predict_x0_from_objective(self, t, x_t, x_T, noise):
        device = x_t.device
        self._move_buffers_to(device)

        m_t = self.m_t[t].view(-1, *([1] * (x_t.dim() - 1)))
        sigma_t = self.variance_t[t].sqrt().view(-1, *([1] * (x_t.dim() - 1)))

        if self.objective_type == "noise":
            x_prev = (x_t - m_t * x_T - sigma_t * noise) / (1. - m_t)
        elif self.objective_type == "grad":
            x_prev = x_t - noise
        elif self.objective_type == "ysubx":
            x_prev = x_T - noise
        return x_prev

    def backward_diffusion(self, i, x_t, x_T, pred_noise):
        device = x_t.device
        self._move_buffers_to(device)

        t = self.steps[i].expand(x_t.size(0))  # shape: [batch]
        if self.steps[i] == 0:
            x0_recon = self.predict_x0_from_objective(t, x_t, x_T, noise=pred_noise)
            return x0_recon, x0_recon
        else:
            n_t = self.steps[i + 1].expand(x_t.size(0))

            x0_recon = self.predict_x0_from_objective(t, x_t, x_T, noise=pred_noise)

            m_t = self.m_t[t].view(-1, *([1] * (x_t.dim() - 1)))
            m_nt = self.m_t[n_t].view(-1, *([1] * (x_t.dim() - 1)))
            var_t = self.variance_t[t].view(-1, *([1] * (x_t.dim() - 1)))
            var_nt = self.variance_t[n_t].view(-1, *([1] * (x_t.dim() - 1)))

            sigma2_t = (var_t - var_nt * (1. - m_t).pow(2) / (1. - m_nt).pow(2)) * var_nt / var_t
            sigma_t = sigma2_t.sqrt() * self.eta

            noise = torch.randn(x_t.shape, generator=self.gen_backward_diffusion).to(device)

            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * x_T + \
                            ((var_nt - sigma2_t) / var_t).sqrt() * (x_t - (1. - m_t) * x0_recon - m_t * x_T)
            return x_tminus_mean + sigma_t * noise, x0_recon

    def p_sample_loop(self, x_T, pred_noise_list):
        device = x_T.device
        self._move_buffers_to(device)

        img = x_T.clone()
        for i in tqdm(range(len(self.steps)), desc='sampling loop'):
            img, x0_recon = self.backward_diffusion(i=i, x_t=img, x_T=x_T, pred_noise=pred_noise_list[i])
        return img, x0_recon

    def q_sample_loop(self, x0, x_T):
        device = x0.device
        self._move_buffers_to(device)

        imgs = [x0]
        objectives = []

        gen_forward_diffusion = torch.Generator().manual_seed(self.seed)
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop'):
            noise = torch.randn(x0.shape, generator=gen_forward_diffusion).to(device)
            t = torch.full((x0.size(0),), i, dtype=torch.long, device=device)
            img, objective = self.forward_diffusion(t, x0, x_T, noise)
            imgs.append(img)
            objectives.append(objective)
        return imgs, objectives
