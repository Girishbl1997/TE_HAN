# reeval_lstm_td3_baseline.py  — behavior-neutral corrected-baseline eval
import os, sys, random
import numpy as np
import torch as th

from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from custom_modules.evaluate import evaluate_policy
from train_manager_lstm_td3 import kill_carla

EVAL_SEED  = 12345
N_EPISODES = 20
SEEDS = [1, 2, 3, 4, 5]
CKPT_NAME  ="D:/Official/TU Dresden/Master Thesis/02_Implementation/TE_AN/models/models_LSTM_TD3/seed_{seed}/best_model.pth"          # swap to "best_model.pth" for peak-SR table)

    
if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())

    c   = ConfigFile("configs/carla_dict_td3.yaml")
    env = CarEnv(seed=EVAL_SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)

    # never sampled during eval -> shrink 100000 -> 1000 (avoids ~20 GB alloc)
    c_proxy._flat_config["buffer_transitions"] = 1000

    agent = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")
    max_steps = SECONDS_PER_EPISODE * 20     # 700

    rows = []
    for s in SEEDS:
        p = os.path.join(Base, "models", "models_LSTM_TD3",
                         f"seed_{s}", CKPT_NAME.format(seed=s))
        if not os.path.exists(p):
            print(f"[WARN] missing {p} -- skipping seed {s}"); continue
        ck = th.load(p, map_location=agent.device)
        agent.actor.load_state_dict(ck["actor"]); agent.critic.load_state_dict(ck["critic"])
        agent.actor.eval(); agent.critic.eval()

        random.seed(EVAL_SEED); np.random.seed(EVAL_SEED)          # identical episodes
        th.manual_seed(EVAL_SEED); th.cuda.manual_seed_all(EVAL_SEED)

        agg = evaluate_policy(agent, env, c_proxy.Env["num_actions"], agent.device,
                              gamma=agent.gamma, n_episodes=N_EPISODES,
                              hidden_size=agent.hidden_size, max_steps=max_steps)
        rows.append((s, agg))
        print(f"[seed {s}] SR={agg['success']:.3f} MCTE={agg['MCTE']:.3f} "
              f"RedLight={agg['red_light']:.3f} Q-MC={agg['Q_minus_MC_bias']:+.2f} "
              f"jerk={agg['jerk']:.4f} dist={agg['distance']:.1f}")

    if rows:
        col = lambda k: np.array([a[k] for _, a in rows])
        print(f"\n==== CORRECTED ARM-1 BASELINE (h_a fixed, n_eps={N_EPISODES}, seeds={len(rows)}) ====")
        for k in ["success","MCTE","red_light","Q_minus_MC_bias","jerk",
                  "CE_steer","distance","collisions_per_km"]:
            v = col(k); print(f"{k:20s} mean={v.mean():+.4f}  std={v.std(ddof=1):.4f}")

    env.close(); 
    kill_carla(port=2000)

