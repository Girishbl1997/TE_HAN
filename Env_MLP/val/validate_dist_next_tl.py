# validate_dist_next_tl.py — trust check for env._next_tl_info() (L1 anticipation signal)
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
L2_SEED   = 1                 # any frozen L2 — we test the ENV signal, not a planner
N_EPS     = 8
MAX_STEPS = SECONDS_PER_EPISODE * 20
MAX_D     = 40.0
TOL       = 2.0               # metres: agreement env-signal vs ground truth
HANDOFF   = 5.0               # metres: jump this big = new (farther) light acquired, not a bug

def force_all_tls_red(world):
    for tl in world.get_actors().filter('*traffic_light*'):
        tl.set_state(carla.TrafficLightState.Red)
        tl.set_red_time(9999.0); tl.set_green_time(0.0); tl.set_yellow_time(0.0)
        tl.freeze(True)

def gt_dist_to_red_ahead(world, vehicle, max_d=MAX_D):
    "Independent ground truth — same geometry as validate_tl_stop_diagnostic."
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
            if (to.x*fwd.x + to.y*fwd.y) <= 0.0: continue
            best = d if (best is None or d < best) else best
    return best

@th.no_grad()
def rollout(agent, env, na, hs, verbose=False):
    obs, _ = env.reset()
    env.world.set_weather(carla.WeatherParameters.ClearNoon)
    force_all_tls_red(env.world)
    env.set_plan(0.0, 0.0)                       # gate=0 so L2 drives naturally (needs the >=0.5 fix!)
    h_a = (th.zeros(1,1,hs,device=agent.device), th.zeros(1,1,hs,device=agent.device))
    prev_a = th.zeros(1,1,na,device=agent.device)
    steps, terminated, truncated = 0, False, False
    prev_env_d = None
    mono_ok, match_ok, redflag_ok, n_zone, max_err = True, True, True, 0, 0.0
    while not (terminated or truncated) and steps < MAX_STEPS + 5:
        if steps % agent.history_length == 0:
            h_a = (th.zeros(1,1,hs,device=agent.device), th.zeros(1,1,hs,device=agent.device))

        # --- signal under test ---
        dist_norm, onehot = env._next_tl_info(max_d=MAX_D)
        env_d = dist_norm * MAX_D
        gt_d  = gt_dist_to_red_ahead(env.world, env.vehicle, MAX_D)
        acquired = (dist_norm < 1.0)            # a TL is ahead & within range

        if acquired:
            n_zone += 1
            if gt_d is not None:                # (a) agrees with independent geometry
                err = abs(env_d - gt_d); max_err = max(max_err, err)
                if err > TOL: match_ok = False
            if onehot[0] < 0.5: redflag_ok = False   # (b) red one-hot lit (all forced red)
            if prev_env_d is not None:          # (c) monotone while closing on same light
                delta = env_d - prev_env_d
                if delta > HANDOFF:  prev_env_d = env_d; 
                elif delta > 0.5:    mono_ok = False; prev_env_d = env_d
                else:                prev_env_d = env_d
            else:
                prev_env_d = env_d
        else:
            prev_env_d = None                   # between lights → reset monotone track

        if verbose and steps % 10 == 0:
            gd = f"{gt_d:6.1f}" if gt_d is not None else "  None"
            print(f"  t={steps:4d} env_d={env_d:6.1f} gt_d={gd} "
                  f"red={onehot[0]:.0f} kmh={obs['kinematics'][0]*90:5.1f}")

        s = {"image": th.tensor(obs["image"],dtype=th.float32,device=agent.device).unsqueeze(0).unsqueeze(0)/255.0,
             "kinematics": th.tensor(obs["kinematics"],dtype=th.float32,device=agent.device).unsqueeze(0).unsqueeze(0)}
        f = agent.critic.encode(s); a_t,_,h_a = agent.actor(f, h_a, prev_a)
        action = a_t.cpu().numpy().flatten()
        obs,_,terminated,truncated,info = env.step(action)
        prev_a = th.tensor(action,dtype=th.float32,device=agent.device).reshape(1,1,-1)
        steps += 1
    return {"n_zone":n_zone,"mono":mono_ok,"match":match_ok,"redflag":redflag_ok,"max_err":max_err}

if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv)>1 else 2010
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())
    c = ConfigFile("configs/carla_dict_td3.yaml")
    env = CarEnv(seed=EVAL_SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)
    c_proxy._flat_config["buffer_transitions"] = 1000
    agent = LSTMTD3DictAgent(c_proxy, "LSTMTD3DictAgent")
    NA = c_proxy.Env["num_actions"]

    ckpt = os.path.join(Base,"models","models_LSTM_TD3",f"seed_{L2_SEED}",f"final_seed{L2_SEED}.pth")
    ck = th.load(ckpt, map_location=agent.device)
    agent.actor.load_state_dict(ck["actor"]); agent.critic.load_state_dict(ck["critic"])
    agent.actor.eval(); agent.critic.eval()

    print(f"[dist_next_TL] frozen L2 = final_seed{L2_SEED}.pth  (env-signal trust check)\n")
    rows=[]
    for i in range(N_EPS):
        sd=EVAL_SEED+i; random.seed(sd); np.random.seed(sd)
        th.manual_seed(sd); th.cuda.manual_seed_all(sd)
        rows.append(rollout(agent, env, NA, agent.hidden_size, verbose=(i==0)))
    print(f"\n{'ep':>3} {'nZone':>6} {'mono':>5} {'match':>6} {'redOK':>6} {'maxErr(m)':>10}")
    for i,r in enumerate(rows):
        print(f"{i:>3} {r['n_zone']:>6} {str(r['mono']):>5} {str(r['match']):>6} "
              f"{str(r['redflag']):>6} {r['max_err']:>10.2f}")
    seen = [r for r in rows if r['n_zone']>0]
    ok = len(seen)>0 and all(r['mono'] and r['match'] and r['redflag'] for r in seen)
    print(f"\nRESULT: {'PASS - _next_tl_info is trustworthy for L1' if ok else 'FAIL - do NOT train L1 yet'}"
          f"  ({len(seen)}/{N_EPS} eps entered the zone)")
    env.close(); kill_carla(port=2000)
