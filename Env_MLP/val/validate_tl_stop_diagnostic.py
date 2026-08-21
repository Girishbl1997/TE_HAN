# validate_tl_stop_diagnostic.py — does frozen L2 stop at a REAL red light? (vision vs early-gate)
import os, sys, random
import numpy as np
import torch as th
import carla

os.environ.setdefault("SHOW_PREVIEW", "1")
from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from train_manager_lstm_td3 import kill_carla

EVAL_SEED = 12345
SEEDS     = [1, 2, 3, 4, 5]
MODES     = ["natural", "early_gate"]   # gate=0 always | gate=1 when red-TL within D_SET
N_EPS     = 15
D_SET, D_LINE, D_ZONE = 25.0, 3.0, 10.0   # metres: lookahead-set | at-line | approach-zone
MAX_STEPS = SECONDS_PER_EPISODE * 20

def force_all_tls_red(world):
    for tl in world.get_actors().filter('*traffic_light*'):
        tl.set_state(carla.TrafficLightState.Red)
        tl.set_red_time(9999.0); tl.set_green_time(0.0); tl.set_yellow_time(0.0)
        tl.freeze(True)

def dist_to_red_ahead(world, vehicle, max_d=40.0):
    loc = vehicle.get_location(); fwd = vehicle.get_transform().get_forward_vector()
    best = None
    for tl in world.get_actors().filter('*traffic_light*'):
        if tl.get_state() != carla.TrafficLightState.Red: continue
        try: stops = tl.get_stop_waypoints()
        except Exception: stops = []
        for wp in stops:
            d = wp.transform.location.distance(loc)
            if d > max_d: continue
            to = wp.transform.location - loc
            if (to.x*fwd.x + to.y*fwd.y) <= 0.0: continue     # behind ego
            best = d if (best is None or d < best) else best
    return best

@th.no_grad()
def rollout(agent, env, mode, na, hs):
    obs, _ = env.reset()
    env.world.set_weather(carla.WeatherParameters.ClearNoon)
    force_all_tls_red(env.world); env.set_plan(0.0, 0.0)
    h_a = (th.zeros(1,1,hs,device=agent.device), th.zeros(1,1,hs,device=agent.device))
    prev_a = th.zeros(1,1,na,device=agent.device)
    enc, min_zone, at_line = False, None, None
    steps, terminated, truncated, term = 0, False, False, {}
    while not (terminated or truncated) and steps < MAX_STEPS + 5:
        if steps % agent.history_length == 0:
            h_a = (th.zeros(1,1,hs,device=agent.device), th.zeros(1,1,hs,device=agent.device))
        d = dist_to_red_ahead(env.world, env.vehicle, 40.0)
        env.set_plan(0.0, 1.0) if (mode=="early_gate" and d is not None and d < D_SET) \
            else env.set_plan(0.0, 0.0)
        s = {"image": th.tensor(obs["image"],dtype=th.float32,device=agent.device).unsqueeze(0).unsqueeze(0)/255.0,
             "kinematics": th.tensor(obs["kinematics"],dtype=th.float32,device=agent.device).unsqueeze(0).unsqueeze(0)}
        f = agent.critic.encode(s); a_t,_,h_a = agent.actor(f,h_a,prev_a)
        action = a_t.cpu().numpy().flatten()
        obs,_,terminated,truncated,info = env.step(action)
        kmh = info.get("Kmph",0.0); term = info.get("termination_reason",{})
        if d is not None:
            enc = True
            if d < D_ZONE: min_zone = kmh if (min_zone is None or kmh<min_zone) else min_zone
            if d < D_LINE and at_line is None: at_line = kmh
        prev_a = th.tensor(action,dtype=th.float32,device=agent.device).reshape(1,1,-1)
        steps += 1
    return {"enc":enc,"min_zone":min_zone,"at_line":at_line,"viol":bool(term.get("red_light"))}

if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv)>1 else 2010
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())
    c = ConfigFile("configs/carla_dict_td3.yaml")
    env = CarEnv(seed=EVAL_SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)
    c_proxy._flat_config["buffer_transitions"] = 1000
    agent = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")
    NA = c_proxy.Env["num_actions"]
    print(f"{'seed':>4} {'mode':>10} | {'enc':>4} {'minZoneKmh':>10} {'atLineKmh':>9} {'violRate':>8}")
    for S in SEEDS:
        ckpt = os.path.join(Base,"models","models_LSTM_TD3",f"seed_{S}",f"final_seed{S}.pth")
        if not os.path.exists(ckpt): print(f"[WARN] missing {ckpt}"); continue
        ck = th.load(ckpt, map_location=agent.device)
        agent.actor.load_state_dict(ck["actor"]); agent.critic.load_state_dict(ck["critic"])
        agent.actor.eval(); agent.critic.eval()
        for mode in MODES:
            cells=[]
            for i in range(N_EPS):
                sd=EVAL_SEED+i; random.seed(sd); np.random.seed(sd)
                th.manual_seed(sd); th.cuda.manual_seed_all(sd)
                cells.append(rollout(agent, env, mode, NA, agent.hidden_size))
            enc=[r for r in cells if r["enc"]]; n=len(enc)
            mz=np.mean([r["min_zone"] for r in enc if r["min_zone"] is not None]) if n else float('nan')
            al=np.mean([r["at_line"]  for r in enc if r["at_line"]  is not None]) if n else float('nan')
            vr=np.mean([r["viol"] for r in enc]) if n else float('nan')
            print(f"{S:>4} {mode:>10} | {n:>4d} {mz:>10.1f} {al:>9.1f} {vr:>8.2f}")
    env.close(); kill_carla(port=2000)
