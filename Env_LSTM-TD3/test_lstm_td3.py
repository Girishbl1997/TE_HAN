import os, sys
import random

os.environ.setdefault("SHOW_PREVIEW", "1")          
os.environ.setdefault("CARLA_WEATHER", "ClearNoon") 

import numpy as np
import torch as th

from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from custom_modules.evaluate import evaluate_policy


def main():

    SEED  = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    Best_model  = sys.argv[2] if len(sys.argv) > 2 else \
            f"D:/Official/TU Dresden/Master Thesis/02_Implementation/TE_AN/models/models_LSTM_TD3/seed_{SEED}/best_model.pth"
    num_eps  = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    eval_seed = 12345
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    th.manual_seed(eval_seed)

    print(f"[TEST] seed={SEED} "
          f"preview={os.environ['SHOW_PREVIEW']} ckpt={os.path.basename(Best_model)}")

    c        = ConfigFile("configs/carla_dict_td3.yaml")
    env      = CarEnv(seed=SEED)
    c_proxy  = AgentConfigFacade(c, env)
    agent    = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")

    # load weights 
    ckpt = th.load(Best_model, map_location=agent.device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval(); agent.critic.eval()
    print("[TEST] weights loaded.")

    # aggregate metrics 
    agg = evaluate_policy(agent, env, c_proxy.Env["num_actions"], agent.device,
                          gamma=agent.gamma, n_episodes=num_eps,
                          hidden_size=agent.hidden_size,
                          max_steps=SECONDS_PER_EPISODE * 20)

    print(f"  SuccessRate (dist>=50 & infraction-free) : {agg['success']:.3f}")
    print(f"  MCTE (moving-only, /lane_width)          : {agg['MCTE']:.4f}")
    print(f"  ControlEffort (mean|steer|)              : {agg['CE_steer']:.4f}")
    print(f"  Jerk (mean|dsteer|)                      : {agg['jerk']:.4f}")
    print(f"  Distance/ep (m)                          : {agg['distance']:.2f}")
    print(f"  Return (scaled)                          : {agg['ret_scaled']:.2f}")
    print(f"  collisions_per_km                        : {agg['collisions_per_km']:.3f}")
    print(f"  RedLightViol / FatalLane                 : {agg['red_light']:.2f} / {agg['fatal_lane']:.2f}")
    print(f"  Q_minus_MC_bias                          : {agg['Q_minus_MC_bias']:+.3f}")

    env.close()

if __name__ == "__main__":
    main()
