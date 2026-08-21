import os
import sys
import time
import random
import numpy as np, torch as th
from torch.utils.tensorboard import SummaryWriter

from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from custom_modules.L1_buffer import L1ReplayBuffer
from custom_modules.L1_agent import L1TD3Agent
from custom_modules.L1_reward import L1Reward
from custom_modules.hier_loop import run_hier_episode
from train_manager_lstm_td3 import kill_carla

OBS_DIM = 11
ACT_DIM = 2
K = 5
SL = 16
WARMUP_EP = 20 # 20
TOTAL_EP = 600 # 600
EXPLORE_NOISE = 0.1
MAX_STEPS = SECONDS_PER_EPISODE * 20
SAVE_EVERY = 25 # 25
 
if __name__ == "__main__":

    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())
    outdir = os.path.join(Base, "models", "models_L1_MLP", f"seed_{SEED}")
    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(Base, "logs", "logs_L1_MLP", f"seed_{SEED}")
    os.makedirs(logdir, exist_ok=True)
    L2_ckpt = os.path.join(Base, "models", "models_LSTM_TD3", f"seed_{SEED}", f"final_seed{SEED}.pth")

    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.cuda.manual_seed_all(SEED)

    c = ConfigFile("configs/carla_dict_td3.yaml")
    env = CarEnv(seed=SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)
    c_proxy._flat_config["buffer_transitions"] = 1000 

    # L2 (forward-only)
    L2 = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")
    ck = th.load(L2_ckpt, map_location=L2.device)
    L2.actor.load_state_dict(ck["actor"])
    L2.critic.load_state_dict(ck["critic"])
    L2.actor.eval()
    L2.critic.eval()

    for p in L2.actor.parameters():
        p.requires_grad = False
    for p in L2.critic.parameters():
        p.requires_grad = False

    # L1 (learned)
    L1 = L1TD3Agent(OBS_DIM, ACT_DIM, c_proxy.Agent, backbone="mlp")
    buf = L1ReplayBuffer(OBS_DIM, ACT_DIM, max_size=200_000)
    L1_reward = L1Reward()
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tb"))

    # Resume
    start_ep = 0
    resume = os.path.join(outdir, "resume.pth")

    if os.path.exists(resume):
        r = th.load(resume, map_location=L1.device)
        L1.load_state_dict(r["L1"])
        start_ep = r["ep"]
        random.setstate(r["py_rng"])
        np.random.set_state(r["np_rng"])
        th.set_rng_state(r["th_rng"].cpu())

        if os.path.exists(os.path.join(outdir, "buffer.pth")):
            buf.load(os.path.join(outdir, "buffer.pth"))
            print(f"[resume] seed {SEED} from ep {start_ep}")

    print(f"[L1-MLP] seed {SEED} | frozen L2 = final_seed{SEED}.pth | {start_ep}->{TOTAL_EP} eps")
    for ep in range(start_ep, TOTAL_EP):
        warm = ep < WARMUP_EP
        ep_r, info = run_hier_episode(env, L2, L1, buf, L1_reward, k=K, SL=SL,
                                      max_steps=MAX_STEPS, explore_noise=EXPLORE_NOISE,
                                      train=True, warmup=warm)

        if not warm and buf.size > L1.batch_size:
            for _ in range(MAX_STEPS // K):
                L1.train(buf)
        writer.add_scalar("L1/EpReward", ep_r, ep)
        writer.add_scalar("L1/NetDisp", info.get("distance", 0.0), ep)
        writer.add_scalar("L1/RedLightViol", int(info.get("termination_reason", {}).get("red_light", False)), ep)

        if (ep + 1) % SAVE_EVERY == 0:
            th.save({"L1": L1.state_dict(), "ep": ep + 1,
                     "py_rng": random.getstate(), 
                     "np_rng": np.random.get_state(),
                     "th_rng": th.get_rng_state()}, resume)
            buf.save(os.path.join(outdir, "buffer.pth"))
            print(f"  [ckpt] ep {ep+1} | epR={ep_r:+.2f}")

    th.save({"actor": L1.actor.state_dict(), "critic": L1.critic.state_dict()},
            os.path.join(outdir, f"final_seed{SEED}.pth"))
    env.close()
    kill_carla(port=2000)
    print(f"[L1-MLP] seed {SEED} done.")