"""Actor Gaussiano + Critico V(s) en PyTorch, y las reglas de actualizacion A2C y PPO-clip.

El actor trabaja en un espacio normalizado u in R^2:
    u ~ N(mu(s), sigma),  mu(s) = tanh(f_theta(s)) in [-1, 1]^2,  sigma = exp(log_std) (parametro)
y el entorno recibe clip(u, -1, 1) re-escalado a v in [0, 0.22] m/s, w in [-2.84, 2.84] rad/s
(env.norm_to_cmd).  log pi(u|s) se calcula sobre u sin recortar.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.5


def mlp(inp, out, hidden=(256, 256)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def _ortho_init(net, last_gain):
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, gain=last_gain)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim=28, act_dim=2, hidden=(256, 256), log_std_init=-0.5):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, tuple(hidden)
        self.mu_net = mlp(obs_dim, act_dim, hidden)
        _ortho_init(self.mu_net, 0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, obs):
        mu = torch.tanh(self.mu_net(obs))
        std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().expand_as(mu)
        return Normal(mu, std)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dist = self.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        u = dist.mean if deterministic else dist.sample()
        return u.squeeze(0).numpy(), dist.log_prob(u).sum(-1).item()

    def config(self):
        return {'obs_dim': self.obs_dim, 'act_dim': self.act_dim, 'hidden': list(self.hidden)}


class Critic(nn.Module):
    def __init__(self, obs_dim=28, hidden=(256, 256)):
        super().__init__()
        self.v_net = mlp(obs_dim, 1, hidden)
        _ortho_init(self.v_net, 1.0)

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)


def save_actor(actor, path, **meta):
    torch.save({'state_dict': actor.state_dict(), 'config': actor.config(), 'meta': meta}, path)


def load_actor(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    actor = GaussianActor(cfg['obs_dim'], cfg['act_dim'], tuple(cfg['hidden']))
    actor.load_state_dict(ck['state_dict'])
    actor.eval()
    return actor, ck.get('meta', {})


# ---------------------------------------------------------------------------- rollouts
class RolloutBuffer:
    def __init__(self):
        self.obs, self.u, self.logp, self.rew, self.done, self.val = [], [], [], [], [], []

    def add(self, obs, u, logp, rew, done, val):
        self.obs.append(obs)
        self.u.append(u)
        self.logp.append(logp)
        self.rew.append(rew)
        self.done.append(done)
        self.val.append(val)

    def __len__(self):
        return len(self.rew)

    def compute(self, last_val, gamma, lam):
        """GAE(lambda).  done[t]=1 corta el bootstrap (los truncamientos ya sumaron gamma*V(s'))."""
        rew = np.asarray(self.rew, dtype=np.float32)
        done = np.asarray(self.done, dtype=np.float32)
        val = np.asarray(self.val + [last_val], dtype=np.float32)
        adv = np.zeros_like(rew)
        gae = 0.0
        for t in reversed(range(len(rew))):
            nonterm = 1.0 - done[t]
            delta = rew[t] + gamma * val[t + 1] * nonterm - val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + val[:-1]
        T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        return T(self.obs), T(self.u), T(self.logp), T(adv), T(ret)


# ---------------------------------------------------------------------------- updates
class A2C:
    """Advantage Actor-Critic estandar: un paso de gradiente por rollout de n pasos,
    L = -E[log pi(u|s) * A] + c_v * (V - R)^2 - c_e * H."""
    name = 'a2c'

    def __init__(self, actor, critic, lr=3e-4, gamma=0.99, lam=1.0, n_steps=32,
                 vf_coef=0.5, ent_coef=1e-3, max_grad=0.5):
        self.actor, self.critic = actor, critic
        self.opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        self.gamma, self.lam, self.n_steps = gamma, lam, n_steps
        self.vf_coef, self.ent_coef, self.max_grad = vf_coef, ent_coef, max_grad

    def update(self, buf, last_val):
        obs, u, _, adv, ret = buf.compute(last_val, self.gamma, self.lam)
        dist = self.actor(obs)
        logp = dist.log_prob(u).sum(-1)
        pg_loss = -(logp * adv).mean()
        v_loss = ((self.critic(obs) - ret) ** 2).mean()
        ent = dist.entropy().sum(-1).mean()
        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                 self.max_grad)
        self.opt.step()
        return {'pg_loss': pg_loss.item(), 'v_loss': v_loss.item(), 'entropy': ent.item()}


class PPO:
    """PPO con objetivo recortado L^CLIP = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)],
    r_t = pi_theta(u|s) / pi_theta_old(u|s), eps = 0.2, varias epocas por minibatches."""
    name = 'ppo'

    def __init__(self, actor, critic, lr=3e-4, gamma=0.99, lam=0.95, n_steps=1024, epochs=10,
                 minibatch=64, clip_eps=0.2, vf_coef=0.5, ent_coef=1e-3, max_grad=0.5):
        self.actor, self.critic = actor, critic
        self.opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        self.gamma, self.lam, self.n_steps = gamma, lam, n_steps
        self.epochs, self.minibatch, self.clip_eps = epochs, minibatch, clip_eps
        self.vf_coef, self.ent_coef, self.max_grad = vf_coef, ent_coef, max_grad

    def update(self, buf, last_val):
        obs, u, logp_old, adv, ret = buf.compute(last_val, self.gamma, self.lam)
        n = len(ret)
        stats = {'pg_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0, 'approx_kl': 0.0}
        k = 0
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.minibatch):
                idx = perm[i:i + self.minibatch]
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                dist = self.actor(obs[idx])
                logp = dist.log_prob(u[idx]).sum(-1)
                ratio = torch.exp(logp - logp_old[idx])
                surr = torch.min(ratio * a,
                                 ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * a)
                pg_loss = -surr.mean()
                v_loss = ((self.critic(obs[idx]) - ret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad)
                self.opt.step()
                with torch.no_grad():
                    stats['pg_loss'] += pg_loss.item()
                    stats['v_loss'] += v_loss.item()
                    stats['entropy'] += ent.item()
                    stats['clip_frac'] += ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                    stats['approx_kl'] += (logp_old[idx] - logp).mean().item()
                k += 1
        return {key: val / max(k, 1) for key, val in stats.items()}
