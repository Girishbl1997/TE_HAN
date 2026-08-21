from collections import deque
import numpy as np 
import torch as th


def _L2_reset_state(L2):
    z = lambda: th.zeros(1, 1, L2.hidden_size, device=L2.device)
    return(z(), z()), th.zeros(1, 1, L2.num_actions, device=L2.device)

@th.no_grad()
def run_hier_episode(env, L2, L1, buffer, L1_reward, k=5, SL=16, 
                     max_steps=700, explore_noise=0.0, train=True, warmup=False):
    obs, _ = env.reset()
    L1_reward.reset()
    h_a, prev_a = _L2_reset_state(L2)
    win = deque([env.get_L1_features() for _ in range(SL)], maxlen=SL)
    env.set_plan(0.0, 0.0)

    steps, ep_r = 0, 0.0
    o_prev = np.stack(win)
    a1 = np.zeros(2, np.float32)
    acc_r = 0.0
    off_cmd, gate_cmd = 0.0, 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated) and steps < max_steps + 5:
        if steps % k == 0:
            o_prev = win[-1].copy()
            a1 = (np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                  if warmup else L1.act(np.stack(win), noise=explore_noise))
            off_cmd = float(np.clip(a1[0], -1.0, 1.0))
            gate_cmd = 0.5*(float(a1[1]) + 1.0)
            env.set_plan(off_cmd, gate_cmd)
            acc_r = 0.0 

        if steps % L2.history_length == 0:
            h_a, _ = _L2_reset_state(L2)
        s = {"image": th.tensor(obs["image"], dtype=th.float32, 
                                device=L2.device).unsqueeze(0).unsqueeze(0)/255.0,
             "kinematics": th.tensor(obs["kinematics"], dtype=th.float32, 
                                     device=L2.device).unsqueeze(0).unsqueeze(0)}
        f = L2.critic.encode(s)
        a_t, _, h_a = L2.actor(f, h_a, prev_a)
        action = a_t.cpu().numpy().flatten()
        obs, _, terminated, truncated, info = env.step(action)
        prev_a = th.tensor(action, dtype=th.float32, device=L2.device).reshape(1, 1, -1)

        win.append(env.get_L1_features())
        acc_r += L1_reward.step(env, off_cmd, gate_cmd)

        if (steps + 1) % k == 0 or terminated or truncated: 
            if info.get("termination_reason", {}).get("red_light", False):
                acc_r = -1.0                

            o_next = win[-1].copy()
            if train:
                buffer.add(o_prev, a1.astype(np.float32), acc_r, o_next, float(terminated))
            ep_r += acc_r

        steps += 1

    if train:
        buffer.commit_episode()
    return ep_r, info