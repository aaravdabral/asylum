# Denoising Diffusion Policy Optimization so its used during RLHF(Reinforcement Learning via Human Feedback , also during diffusion models for the CNN and CV the main task for this is the brownian motion occuring in the pixels of the image tokens/embeddings and the denoising is done via the diffusion model and the policy optimization is done via the PPO algorithm     )

import math
import torch
import torch.nn as nn

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


T = 50
betas = torch.linspace(1e-3, 0.2, T, device=device)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)
sigmas = betas.sqrt()          



class Denoiser(nn.Module):
    def __init__(self, hidden=128, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)
        self.net = nn.Sequential(
            nn.Linear(2 + 2 * n_freq, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):                       
        tt = (t.float() / T).unsqueeze(-1) * self.freqs
        return self.net(torch.cat([x, tt.sin(), tt.cos()], dim=-1))


def sample_data(n):
    
    centers = torch.tensor([[2., 2.], [2., -2.], [-2., 2.], [-2., -2.]], device=device)
    return centers[torch.randint(0, 4, (n,), device=device)] + 0.2 * torch.randn(n, 2, device=device)


def reward_fn(x0):
    
    return (x0[:, 0] > 0).float()



def step_mean(model, x, t):
    eps = model(x, t)
    return (x - betas[t].unsqueeze(-1) / (1 - abar[t]).sqrt().unsqueeze(-1) * eps) / alphas[t].sqrt().unsqueeze(-1)


def gaussian_logprob(x_prev, mean, sigma):
    var = sigma ** 2
    lp = -((x_prev - mean) ** 2) / (2 * var) - sigma.log() - 0.5 * math.log(2 * math.pi)
    return lp.sum(-1)                              



model = Denoiser().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for it in range(3000):
    x0 = sample_data(512)
    t = torch.randint(0, T, (512,), device=device)
    eps = torch.randn_like(x0)
    xt = abar[t].sqrt().unsqueeze(-1) * x0 + (1 - abar[t]).sqrt().unsqueeze(-1) * eps
    loss = ((model(xt, t) - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        print(f"pretrain {it} loss {loss.item():.4f}")



@torch.no_grad()
def sample_trajectories(model, n):
    x = torch.randn(n, 2, device=device)
    xs, xs_prev, lps = [], [], []
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        mean = step_mean(model, x, tt)
        x_prev = mean + sigmas[t] * torch.randn_like(x)
        xs.append(x); xs_prev.append(x_prev)
        lps.append(gaussian_logprob(x_prev, mean, sigmas[t]))
        x = x_prev
    
    return torch.stack(xs, 1), torch.stack(xs_prev, 1), torch.stack(lps, 1), x


N, ITERS, INNER_EPOCHS, LR, CLIP = 256, 40, 2, 3e-4, 0.2
opt = torch.optim.Adam(model.parameters(), lr=LR)

print(f"before DDPO: {reward_fn(sample_trajectories(model, 2000)[3]).mean():.3f} of samples on the right")

for it in range(ITERS):
    xs, xs_prev, old_lp, x0 = sample_trajectories(model, N)
    r = reward_fn(x0)
    adv = (r - r.mean()) / (r.std() + 1e-8)        # one advantage per sample, shared by all its timesteps

    for _ in range(INNER_EPOCHS):
        loss = 0.0
        for k in range(T):
            t = T - 1 - k
            tt = torch.full((N,), t, device=device, dtype=torch.long)
            new_lp = gaussian_logprob(xs_prev[:, k], step_mean(model, xs[:, k], tt), sigmas[t])
            ratio = torch.exp(new_lp - old_lp[:, k])
            loss = loss + torch.max(-adv * ratio, -adv * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
        opt.zero_grad(); (loss / T).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    if it % 5 == 0:
        print(f"iter {it} mean reward {r.mean():.3f}")

print(f"after DDPO:  {reward_fn(sample_trajectories(model, 2000)[3]).mean():.3f} of samples on the right")