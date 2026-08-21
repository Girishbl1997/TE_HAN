import numpy as np
import torch

class L1ReplayBuffer:
    "Flat Low-dim planner: 7 + 4 obs, no images, no recurrence"
    def __init__(self, obs_dim, action_dim, max_size, device="cuda"):
        self.capacity, self.obs_dim, self.action_dim = int(max_size), int(obs_dim), int(action_dim)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        z = lambda d: torch.zeros((self.capacity, d), dtype=torch.float32, device=self.device)
        self.obs, self.next = z(obs_dim), z(obs_dim)
        self.act = z(action_dim)
        self.rew = z(1)
        self.done = z(1)
        self.ep_id = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)
        self.ptr, self.size, self.cur_ep = 0, 0, 0

    def add(self, s, a, r, s2, d):
        p = self.ptr
        self.obs[p] = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        self.next[p] = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        self.act[p] = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.rew[p] = float(r)
        self.done[p] = float(d)
        self.ep_id[p] = self.cur_ep
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def commit_episode(self):
        self.cur_ep += 1

    def _idx(self, b):
        return torch.randint(0, self.size, (b,), device=self.device)
    
    def sample(self, b):
        i = self._idx(b)
        return self.obs[i], self.act[i], self.rew[i], self.next[i], self.done[i]

    def sample_windows(self, b, L):
        i = self._idx(b)
        offs = torch.arange(L, device=self.device)
        gather = (i[:, None] - (L - 1) + offs[None, :]) % self.capacity 
        same_ep = (self.ep_id[gather] == self.ep_id[i][:, None]).unsqueeze(-1)

        obs_win = self.obs[gather] * same_ep 
        next_win = self.next[gather] * same_ep 
        return obs_win, self.act[i], self.rew[i], next_win, self.done[i]

    def save(self, path):
        torch.save({"obs": self.obs, "next": self.next, "act": self.act, "rew": self.rew,
                    "done": self.done, "ep_id": self.ep_id, "ptr": self.ptr, 
                    "size": self.size, "cur_ep": self.cur_ep}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        for k in ("obs", "next", "act", "rew", "done", "ep_id"):
            getattr(self, k).copy_(d[k])
        self.ptr, self.size, self.cur_ep = d["ptr"], d["size"], d["cur_ep"]
