# validate_plan_sweep.py — Phase-1 command-channel proof + L2 stop-responsiveness screen
import os, sys, random
import numpy as np
import torch as th
import carla

os.environ.setdefault("SHOW_PREVIEW", "1")   # set "1" only for the one-off sign eyeball

from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from train_manager_lstm_td3 import kill_carla

EVAL_SEED    = 12345
SEEDS        = [1, 2, 3, 4, 5]          # screens every frozen L2 in one CARLA session
OFFSETS      = [-1.0, -0.5, 0.0, 0.5, 1.0]
GATES        = [0, 1]
EPS_PER_CELL = 10                        # was 3; paired across cells
TAIL         = 32                        # steady-state window (last N steps)
MAX_STEPS    = SECONDS_PER_EPISODE * 20  # 700

@th.no_grad()
def rollout(agent, env, plan, num_actions, hidden_size):
    """One episode, CONSTANT plan. Caller seeds RNG BEFORE this (paired spawns)."""
    obs, _ = env.reset()
    env.world.set_weather(carla.WeatherParameters.ClearNoon)   # PIN weather (env ignores CARLA_WEATHER)
    env.set_plan(*plan)                                          # re-latch AFTER reset (reset zeroes it)

    h_a = (th.zeros(1,1,hidden_size, device=agent.device),
           th.zeros(1,1,hidden_size, device=agent.device))
    prev_a = th.zeros(1,1,num_actions, device=agent.device)

    kmhs, signed, moving = [], [], []
    steps, terminated, truncated, term = 0, False, False, {}
    a_equiv_ok = True
    is_baseline = (abs(plan[0]) < 1e-9 and plan[1] < 0.5)

    while not (terminated or truncated) and steps < MAX_STEPS + 5:
        if steps % agent.history_length == 0:                   # h_a reset every seq_len=16
            h_a = (th.zeros(1,1,hidden_size, device=agent.device),
                   th.zeros(1,1,hidden_size, device=agent.device))
        s = {"image": th.tensor(obs["image"], dtype=th.float32,
                                device=agent.device).unsqueeze(0).unsqueeze(0)/255.0,
             "kinematics": th.tensor(obs["kinematics"], dtype=th.float32,
                                     device=agent.device).unsqueeze(0).unsqueeze(0)}
        f = agent.critic.encode(s)
        a_t, _, h_a = agent.actor(f, h_a, prev_a)               # deterministic
        action = a_t.cpu().numpy().flatten()
        obs, _, terminated, truncated, info = env.step(action)

        kmh_i = info.get("Kmph", 0.0)
        sdev  = info.get("signed_wp_dev", 0.0)
        kmhs.append(kmh_i); signed.append(sdev)
        moving.append(kmh_i > 1.0)
        term = info.get("termination_reason", {})

        # (iv) A-equivalence: at offset=0, injected signed dev must equal true wp_dev
        if is_baseline and abs(abs(sdev) - info.get("wp_dev", 0.0)) > 1e-3:
            a_equiv_ok = False

        prev_a = th.tensor(action, dtype=th.float32,
                           device=agent.device).reshape(1,1,-1)
        steps += 1

    flags = ["collision","fatal_lane","lock","red_light","wp_dev_exceeded"]
    any_flag = any(bool(term.get(k)) for k in flags)
    silent_stationary = terminated and (not truncated) and (not any_flag) and steps < MAX_STEPS
    success = float(info.get("distance",0.0) >= 50.0 and not any_flag)

    kmhs, signed, moving = np.array(kmhs), np.array(signed), np.array(moving)
    tail_kmh = float(kmhs[-TAIL:].mean()) if len(kmhs) else 0.0
    tail_mv  = signed[-TAIL:][moving[-TAIL:]] if moving[-TAIL:].any() else signed[-TAIL:]
    tail_signed = float(tail_mv.mean()) if len(tail_mv) else 0.0

    return {"len": steps, "tail_signed": tail_signed, "tail_kmh": tail_kmh,
            "success": success, "silent_stationary": float(silent_stationary),
            "wp_exc": float(bool(term.get("wp_dev_exceeded"))),
            "a_equiv": float(a_equiv_ok)}

if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 2010
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())

    c = ConfigFile("configs/carla_dict_td3.yaml")
    env = CarEnv(seed=EVAL_SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)
    c_proxy._flat_config["buffer_transitions"] = 1000          # avoid ~20 GB alloc
    agent = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")
    NA = c_proxy.Env["num_actions"]

    R = {}   # (seed, off, gate) -> aggregated metrics
    for S in SEEDS:
        ckpt = os.path.join(Base, "models", "models_LSTM_TD3",
                            f"seed_{S}", f"final_seed{S}.pth")   # final_seed rule
        if not os.path.exists(ckpt):
            print(f"[WARN] missing {ckpt} — skip seed {S}"); continue
        ck = th.load(ckpt, map_location=agent.device)
        agent.actor.load_state_dict(ck["actor"]); agent.critic.load_state_dict(ck["critic"])
        agent.actor.eval(); agent.critic.eval()
        print(f"\n[SWEEP] frozen L2 = {os.path.basename(ckpt)}")
        print(f"{'offset':>7} {'gate':>4} | {'tailSgn':>8} {'tailKmh':>8} {'len':>5} "
              f"{'SR':>4} {'wpExc':>5} {'silent':>7} {'Aeq':>4}")
        for gate in GATES:
            for off in OFFSETS:
                cell = []
                for i in range(EPS_PER_CELL):                   # (ii) paired seeding per episode index
                    sd = EVAL_SEED + i
                    random.seed(sd); np.random.seed(sd)
                    th.manual_seed(sd); th.cuda.manual_seed_all(sd)
                    cell.append(rollout(agent, env, (off, gate), NA, agent.hidden_size))
                m = lambda k: float(np.mean([r[k] for r in cell]))
                R[(S,off,gate)] = {k: m(k) for k in cell[0]}
                sr = m('success') if gate == 0 else float('nan')   # SR meaningless when commanded to stop
                print(f"{off:>7.2f} {gate:>4d} | {m('tail_signed'):>8.3f} {m('tail_kmh'):>8.1f} "
                      f"{m('len'):>5.0f} {sr:>4.2f} {m('wp_exc'):>5.2f} "
                      f"{m('silent_stationary'):>7.2f} {m('a_equiv'):>4.2f}")

    # ---- (v) cross-seed screens ----
    print("\n================  L2 STOP-RESPONSIVENESS SCREEN  ================")
    print(f"{'seed':>4} | {'drive_kmh(0,g0)':>15} {'stop_kmh(0,g1)':>15} {'gain':>6} {'responsive':>11}")
    for S in SEEDS:
        if (S,0.0,0) not in R: continue
        drv, stp = R[(S,0.0,0)]['tail_kmh'], R[(S,0.0,1)]['tail_kmh']
        print(f"{S:>4} | {drv:>15.1f} {stp:>15.1f} {drv-stp:>6.1f} "
              f"{'YES' if stp < 2.0 else 'NO':>11}")

    print("\n================  OFFSET-TRACKING SCREEN (gate=0)  ================")
    print(f"{'seed':>4} | {'sgn@-1':>7} {'sgn@0':>6} {'sgn@+1':>7} {'sign_ok':>7} {'mono_ok':>7}")
    for S in SEEDS:
        if (S,-1.0,0) not in R: continue
        n1, z0, p1 = (R[(S,-1.0,0)]['tail_signed'], R[(S,0.0,0)]['tail_signed'],
                      R[(S, 1.0,0)]['tail_signed'])
        sign_ok = (p1 > 0.15) and (n1 < -0.15)
        mono_ok = (n1 < z0 < p1)
        print(f"{S:>4} | {n1:>7.3f} {z0:>6.3f} {p1:>7.3f} "
              f"{'YES' if sign_ok else 'NO':>7} {'YES' if mono_ok else 'NO':>7}")

    env.close(); kill_carla(port=2000)
