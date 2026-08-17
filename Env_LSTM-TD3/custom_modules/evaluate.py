
import numpy as np
import torch as th


@th.no_grad()
def evaluate_policy(agent, env, num_actions, device, gamma=0.99,
                    n_episodes=10, lane_width=3.5, hidden_size=256,
                    max_steps=700):

    ep_metrics = []
    all_Qpred, all_MC = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        h_a  = (th.zeros(1, 1, hidden_size, device=device),
                th.zeros(1, 1, hidden_size, device=device))
        
        prev_a = th.zeros(1, 1, num_actions, device=device)

        rewards, q1_preds = [], []
        wp_devs, steers, jerks = [], [], []
        term_reason = {}
        done, steps = False, 0

        while not done and steps < max_steps + 5:
            if steps % agent.history_length == 0:      # history_len = seq_len = 16
                 h_c1 = (th.zeros(1, 1, hidden_size, device=device),
                        th.zeros(1, 1, hidden_size, device=device))
            s = {"image": th.tensor(obs["image"], dtype=th.float32,
                                    device=device).unsqueeze(0).unsqueeze(0) / 255.0,
                 "kinematics": th.tensor(obs["kinematics"], dtype=th.float32,
                                         device=device).unsqueeze(0).unsqueeze(0)}

            f = agent.critic.encode(s)
            a_t, _, h_a = agent.actor(f, h_a, prev_a)             # deterministic
            q1, h_c1 = agent.critic.single_forward(f, a_t, h_c1, prev_a)
            q1_preds.append(q1.item())

            action = a_t.cpu().numpy().flatten()
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(r)                                 # scaled
            if info.get("Kmph", 0.0) > 1.0:
                wp_devs.append(info.get("wp_dev", 0.0))
            steers.append(abs(float(action[0])))
            jerks.append(info.get("steer_diff", 0.0))
            term_reason = info.get("termination_reason", {})

            prev_a = th.tensor(action, dtype=th.float32,
                               device=device).reshape(1, 1, -1)
            steps += 1

        # Monte-Carlo discounted return per visited state
        G, mc = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            mc.append(G)
        mc.reverse()
        all_Qpred.extend(q1_preds)
        all_MC.extend(mc)

        success = (float(info.get("distance", 0.0 )) >= 50.0 
                   and not (term_reason.get("collision") or term_reason.get("fatal_lane")
                    or term_reason.get("lock") or term_reason.get("red_light")
                    or term_reason.get("wp_dev_exceeded")))

        ep_metrics.append({
            "ret_scaled": float(np.sum(rewards)),
            "length":     steps,
            "distance":   float(info.get("distance", 0.0)),
            "success":    float(success),
            "collision":  float(bool(term_reason.get("collision"))),
            "fatal_lane": float(bool(term_reason.get("fatal_lane"))),
            "red_light":  float(bool(term_reason.get("red_light"))),
            "MCTE":       (float(np.mean(wp_devs)) / lane_width) if wp_devs else 0.0,
            "CE_steer":   float(np.mean(steers)),
            "jerk":       float(np.mean(jerks)),
        })

    agg = {k: float(np.mean([m[k] for m in ep_metrics])) for k in ep_metrics[0]}
    dist_km = max(sum(m["distance"] for m in ep_metrics) / 1000.0, 1e-6)
    agg["collisions_per_km"] = sum(m["collision"] for m in ep_metrics) / dist_km

    Qp, MC = np.array(all_Qpred), np.array(all_MC)
    agg["Q_pred_mean"]     = float(Qp.mean())
    agg["MC_return_mean"]  = float(MC.mean())
    agg["Q_minus_MC_bias"] = float(Qp.mean() - MC.mean())   # overestimation proof
    return agg


def log_eval(agent, agg, global_step):
    w = agent.writer
    w.add_scalar("Eval/TestReturn_scaled", agg["ret_scaled"],       global_step)
    w.add_scalar("Eval/SuccessRate",       agg["success"],          global_step)
    w.add_scalar("Eval/Collisions_per_km", agg["collisions_per_km"],global_step)
    w.add_scalar("Eval/MCTE",              agg["MCTE"],             global_step)
    w.add_scalar("Eval/ControlEffort",     agg["CE_steer"],         global_step)
    w.add_scalar("Eval/Jerk",              agg["jerk"],             global_step)
    w.add_scalar("Eval/RedLightViol",      agg["red_light"],        global_step)
    w.add_scalar("Diag/Q_minus_MC_bias",   agg["Q_minus_MC_bias"],  global_step)
    w.add_scalar("Diag/MC_return_mean",    agg["MC_return_mean"],    global_step)
    print(f"[EVAL @ {global_step}] SR={agg['success']:.2f} MCTE={agg['MCTE']:.3f} "
          f"CE={agg['CE_steer']:.3f} jerk={agg['jerk']:.4f} "
          f"Q-MC={agg['Q_minus_MC_bias']:+.3f}")
