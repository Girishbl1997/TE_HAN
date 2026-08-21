import copy
import math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

# Agent B 
class L1_MLP_Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h),
                                 nn.LayerNorm(h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), 
                                 nn.Linear(h, action_dim))
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs[:, -1, :]
        return torch.tanh(self.net(obs))

class L1_MLP_TwinCritic(nn.Module):

    def __init__(self, obs_dim, action_dim, h=128):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(obs_dim + action_dim, h),
                                   nn.LayerNorm(h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU(),
                                   nn.Linear(h, 1))
        self.q1, self.q2 = mk(), mk()

    def _x(self, obs, a):
        if obs.dim() == 3: 
            obs = obs[:, -1, :]
        return torch.cat([obs, a], dim=-1)
    
    def forward(self, obs, a):
        x = self._x(obs, a)
        return self.q1(x), self.q2(x)

    def Q1(self, obs, a):
        return self.q1(self._x(obs, a))

class L1TD3Agent:

    def __init__(self, obs_dim, action_dim, cfg, backbone="mlp", device="cuda"):
        self.device  = torch.device(device if torch.cuda.is_available() else "cpu")
        self.action_dim, self.backbone = action_dim, backbone

        agent_cfg = cfg.get("Agent", cfg)
        self.gamma, self.tau = agent_cfg.get("gamma"), agent_cfg.get("tau")
        self.batch_size = agent_cfg.get("batch_size")
        self.tgt_noise = agent_cfg.get("policy_noise")
        self.tgt_noise_clip = agent_cfg.get("policy_noise_clip")
        self.pol_upd_delay = agent_cfg.get("policy_delay")
        self.CRITIC_WARMUP = agent_cfg.get("critic_warmup", 2000)
        self.L = agent_cfg.get("sequence_length")

        if backbone == "mlp":
            self.actor = L1_MLP_Actor(obs_dim, action_dim).to(self.device)
            self.critic = L1_MLP_TwinCritic(obs_dim, action_dim).to(self.device)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=agent_cfg.get("actor_lr"))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=agent_cfg.get("critic_lr"))
        self.grad_steps = 0
        self.pol_upd_cnt = 0

    @torch.no_grad()
    def act(self, obs_window, noise=0.0):
        x = torch.as_tensor(obs_window, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(x).cpu().numpy().flatten()
        if noise > 0.0:
            a = np.clip(a + np.random.normal(0, noise, size=self.action_dim), -1.0, 1.0)
        return a

    def _sample(self, buf):
        return buf.sample(self.batch_size) if self.backbone == "mlp" \
            else buf.sample_windows(self.batch_size, self.L)

    def train(self, buffer):
        self.grad_steps += 1
        s, a, r, s2, d = self._sample(buffer)

    # Critic
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            a2 = self.target_actor(s2)
            n = torch.clamp(torch.randn_like(a2)*self.tgt_noise, -self.tgt_noise_clip, self.tgt_noise_clip)
            a2 = torch.clamp(a2 + n, -1.0, 1.0)
            q1t, q2t = self.target_critic(s2, a2)
            y = r + self.gamma*(1.0 - d)*torch.min(q1t, q2t)

        q1, q2 = self.critic(s, a)
        (((q1 - y)**2).mean() + ((q2 - y)**2).mean()).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

    # Delayed actor
        if self.grad_steps % self.pol_upd_delay == 0:
            if self.grad_steps > self.CRITIC_WARMUP:
                for p in self.critic.parameters():
                    p.requires_grad = False
                self.actor_optimizer.zero_grad()
                a_pi = self.actor(s)
                (-self.critic.Q1(s, a_pi).mean() + 1e-3*(a_pi**2).mean()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                for p in self.critic.parameters():
                    p.requires_grad = True
                self.pol_upd_cnt += 1
            self._polyak()
        
    def _polyak(self):

        with torch.no_grad():
            for tgt, src in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
                for tp, p in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau*p.data)

    def state_dict(self):
        return{"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "grad_steps": self.grad_steps, "pol_upd_cnt": self.pol_upd_cnt
            }

    def load_state_dict(self, ck):
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.target_critic.load_state_dict(ck["target_critic"])
        self.actor_optimizer.load_state_dict(ck["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ck["critic_optimizer"])
        self.grad_steps = ck["grad_steps"]
        self.pol_upd_cnt = ck["pol_upd_cnt"]