import numpy as np
import torch

class RecurrentDictReplayBuffer:
    """ Flat, Transition indexed for dict obs,
        capacity = transitions, sequences are taken from 1 episode,
        next_obs stored per transition, semantically identical """
    
    def __init__(self, state_shape: dict,action_dim: int, max_size: int, max_episode_length: int):
        self.capacity = int(max_size)
        self.max_steps = int(max_episode_length)
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_shape = tuple(state_shape["image"])
        kin_shape = tuple(state_shape["kinematics"])
        # 
        self.obs_image = torch.zeros((self.capacity, *img_shape), dtype=torch.uint8, device=self.device)
        self.obs_kin = torch.zeros((self.capacity, *kin_shape), dtype=torch.float32, device=self.device)
        self.next_obs_image = torch.zeros((self.capacity, *img_shape), dtype=torch.uint8, device=self.device)
        self.next_obs_kin = torch.zeros((self.capacity, *kin_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
       
        # episode-boundary metadata
        self.max_eps = self.capacity
        self.ep_start = np.zeros((self.max_eps,), dtype=np.int64)
        self.ep_len = np.zeros((self.max_eps,), dtype=np.int64)
        self.ep_head = 0
        self.ep_count = 0

        self.ptr = 0
        self.size = 0
        self.cur_ep_start = 0
        self.current_ep_step = 0
    
    def _evict_if_needed(self, write_pos):
        # remove oldest episode if overwrites
        while self.ep_count > 0 and write_pos == self.ep_start[self.ep_head]:
            self.ep_head = (self.ep_head + 1) % self.max_eps
            self.ep_count -= 1

    def add(self, s: dict, a, r, s2:dict, d):
        if self.current_ep_step >= self.max_steps:
            return
        if self.current_ep_step == 0:
            self.cur_ep_start = self.ptr
        self._evict_if_needed(self.ptr)

        p = self.ptr
        self.obs_image[p] = torch.as_tensor(s["image"], device=self.device)
        self.obs_kin[p] = torch.as_tensor(s["kinematics"], device=self.device)
        self.next_obs_image[p] = torch.as_tensor(s2["image"], device=self.device)
        self.next_obs_kin[p] = torch.as_tensor(s2["kinematics"], device=self.device)
        self.actions[p] = torch.as_tensor(a, device=self.device)
        self.rewards[p] = float(r)
        self.dones[p] = float(d)
       
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        self.current_ep_step += 1
      
    def commit_episode(self):
        if self.current_ep_step == 0:
            return
        if self.ep_count == self.max_eps:
            self.ep_head = (self.ep_head + 1) % self.max_eps
            self.ep_count -= 1

        tail = (self.ep_head + self.ep_count) % self.max_eps
        self.ep_start[tail] = self.cur_ep_start
        self.ep_len[tail] = self.current_ep_step
        self.ep_count += 1
       
        self.current_ep_step = 0

    def sample(self, batch_size, seq_len, burn_in = 0):

        assert self.ep_count > 0, "sample() before any committed eps"
        L = seq_len + burn_in
        img_c = self.obs_image.shape[1:]
        kin_c = self.obs_kin.shape[1:]  

        obs_image_batch = np.zeros((batch_size, L, *img_c), dtype=np.uint8)
        obs_kin_batch = np.zeros((batch_size, L, *kin_c), dtype=np.float32)
        next_obs_image_batch = np.zeros((batch_size, L, *img_c), dtype=np.uint8)
        next_obs_kin_batch = np.zeros((batch_size, L, *kin_c), dtype=np.float32)
        
        act_batch = np.zeros((batch_size, L, self.action_dim), dtype=np.float32)
        rew_batch = np.zeros((batch_size, L, 1), dtype=np.float32)
        done_batch = np.zeros((batch_size, L, 1), dtype=np.float32)
        mask_batch = np.zeros((batch_size, L), dtype=np.float32)

        ep_slots = (self.ep_head + np.random.randint(0, self.ep_count, size=batch_size)) % self.max_eps
        ep_start_v = self.ep_start[ep_slots].astype(np.int64)
        ep_len_v = self.ep_len[ep_slots].astype(np.int64)

        max_off = np.maximum(0, ep_len_v - L)
        start_off = np.floor(np.random.rand(batch_size)*(max_off + 1)).astype(np.int64)
        start_off = np.minimum(start_off, max_off)
        actual = np.minimum(L, ep_len_v - start_off)

        offs = np.arange(L)
        gather = (ep_start_v[:, None] + start_off[:, None] + offs[None, :]) % self.capacity 
        valid = offs[None, :] < actual[:, None]
        inv = ~valid      

        gather_t = torch.as_tensor(gather, device=self.device, dtype=torch.long)
        inv_t = torch.as_tensor(~valid, device=self.device)

        obs_image_batch = self.obs_image[gather_t]; obs_image_batch[inv_t] = 0
        obs_kin_batch = self.obs_kin[gather_t]; obs_kin_batch[inv_t] = 0
        next_obs_image_batch = self.next_obs_image[gather_t]; next_obs_image_batch[inv_t] = 0
        next_obs_kin_batch = self.next_obs_kin[gather_t]; next_obs_kin_batch[inv_t] = 0 
        act_batch = self.actions[gather_t]; act_batch[inv_t] = 0
        rew_batch = self.rewards[gather_t]; rew_batch[inv_t] = 0
        done_batch = self.dones[gather_t]; done_batch[inv_t] = 0

        mask_batch = valid.astype(np.float32)
        if burn_in > 0:
            zero_upto = np.where(actual > burn_in, burn_in, np.maximum(0, actual - 1))
            mask_batch[offs[None, :] < zero_upto[:, None]] = 0.0
        m = torch.as_tensor(mask_batch, device=self.device)

        s = {"image":obs_image_batch.float()/255.0, "kinematics": obs_kin_batch}
        s2 = {"image": next_obs_image_batch.float()/255.0, "kinematics": next_obs_kin_batch}
        a = act_batch
        r = rew_batch
        d = done_batch
        return s, a, r, s2, d, m


