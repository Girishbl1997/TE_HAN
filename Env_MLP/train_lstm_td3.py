
import os
import sys
import time
import numpy as np 
import torch as th
import random
import logging
from torch.utils.tensorboard import SummaryWriter

logging.getLogger("tud_rl").setLevel(logging.WARNING)

from tud_rl.common.configparser import ConfigFile
from custom_modules.config_proxy import AgentConfigFacade
from tud_rl.common.logging_func import EpochLogger

os.environ.setdefault("SHOW_PREVIEW", "1") # change it "0" for long runs

from custom_modules.env_lstm_td3 import CarEnv, SECONDS_PER_EPISODE
from custom_modules.recurrent_dict_buffer import RecurrentDictReplayBuffer
from custom_modules.agent_lstm_td3_dict import LSTMTD3DictAgent
from custom_modules.evaluate import evaluate_policy, log_eval
from train_manager_lstm_td3 import kill_carla

EVAL_EVERY = 25000
th.backends.cudnn.benchmark = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == "__main__": 

    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 2010
    Base = os.environ.get("PROJECT_ROOT", os.getcwd())
    modelsdir = os.path.join(Base, "models", "models_LSTM_TD3", f"seed_{SEED}")
    logsdir = os.path.join(Base, "logs", "logs_LSTM_TD3", f"seed_{SEED}")
    # modelsdir = f"D:/Official/TU Dresden/Master Thesis/02_Implementation/TE_AN/models/models_LSTM_TD3/seed_{SEED}" 
    # logsdir = f"D:/Official/TU Dresden/Master Thesis/02_Implementation/TE_AN/logs/logs_LSTM_TD3/seed_{SEED}"

    print('Initializing phase 2 level 2 native training.....')
    print( 'Creating logs and models...')

    if not os.path.exists(modelsdir):
        os.makedirs(modelsdir)
    if not os.path.exists(logsdir):
        os.makedirs(logsdir)
    
    random.seed(SEED); np.random.seed(SEED)
    th.manual_seed(SEED); th.cuda.manual_seed_all(SEED)

    print("connecting to native config....")
    c = ConfigFile("configs/carla_dict_td3.yaml")
    agent_name = "LSTMTD3DictAgent"
    
    env = CarEnv(seed=SEED, port=PORT)
    c_proxy = AgentConfigFacade(c, env)
    TOTAL_TIMESTEPS=int(c_proxy.total_timesteps)

    #Initialize custom Agent
    agent = LSTMTD3DictAgent(c_proxy, agent_name)
    tb_log_dir = os.path.join(logsdir, "tensorboard")
    agent.writer = SummaryWriter(log_dir = tb_log_dir)
    tabular_dir = os.path.join(logsdir, f"tabular_{int(time.time())}")

    logger = EpochLogger(agent_name, SEED, output_dir = tabular_dir)
    agent.logger = logger
    print("connecting to Env_LSTM-TD3....") 
    #    
    ACT_START_STEP = int(c_proxy.act_start_step)
    UPD_START_STEP = int(c_proxy.upd_start_step)
    UPD_EVERY = int(c_proxy.upd_every)
    G = 32 # Inner gradient loop  UTD = G/UPD_EVERY = 32/100 = 0.32, 

    best_score = (-1.0, -float('inf'))     
    step = 0
    last_eval = 0
    episodes = 0

    obs,_ = env.reset()
    inference_hidden_state = (th.zeros(1, 1, 256).to(agent.device), th.zeros(1, 1, 256).to(agent.device))
    prev_a = np.zeros((1, 1, c_proxy.Env["num_actions"]), dtype=np.float32)

    print(f"Training model on Env_LSTM-TD3 for {TOTAL_TIMESTEPS} timesteps, Seed {SEED}")
    ep_reward = 0.0
    ep_steer_diff = 0.0
    ep_wp_dev = 0.0
    explore_steer = 0.0
    explore_throttle = 0.0
   
    while step < TOTAL_TIMESTEPS:
        dict_s = {
            "image" : th.tensor(obs["image"], dtype=th.float32, device = agent.device).unsqueeze(0).unsqueeze(0)/255.0,
            "kinematics": th.tensor(obs["kinematics"], dtype=th.float32, device = agent.device).unsqueeze(0).unsqueeze(0)
        }
        prev_a_tensor = th.tensor(prev_a, dtype=th.float32, device=agent.device)

        if step < ACT_START_STEP: 
            if step % 20 == 0:
                explore_steer = np.clip(np.random.normal(loc = 0.0, scale = 0.5), -1.0, 1.0)
                explore_throttle = np.random.uniform(low = -0.5, high = 0.8)
            action = np.array([explore_steer, explore_throttle], dtype=np.float32)
        else: 
            with th.no_grad():
                f = agent.critic.encode(dict_s)
                
                action_tensor, _, new_hidden_state = agent.actor(f, inference_hidden_state, prev_a_tensor)
                action = action_tensor.cpu().numpy().flatten()
                inference_hidden_state = new_hidden_state
        
            noise = np.random.normal(0, 0.1, size=c_proxy.Env["num_actions"])
            action = np.clip(action + noise, -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.add(s=obs, a=action, r=reward, s2=next_obs, d=terminated)
        
        obs=next_obs
        prev_a = action.reshape(1, 1, -1)    

        step+= 1
        if step % 25000 == 0:
            th.save({'actor': agent.actor.state_dict(),
                     'critic': agent.critic.state_dict()},
                     os.path.join(modelsdir, f"ckpt_{step}.pth"))

        # Progress Tracking
        if step % 25000 == 0:
            progress_pct = (step/TOTAL_TIMESTEPS) * 100
            print(f" [Seed {SEED}] Progress: {step}/{TOTAL_TIMESTEPS} steps --- {progress_pct:.1f} % | ")
            
            # 1. Log numeric progress to visualize as a curve
            agent.writer.add_scalar("Training/Progress_Pct", progress_pct, global_step=step)

            # 2. (Optional) Log the formatted string to the TensorBoard Text tab
            status_msg = f"[Seed {SEED}] Progress: {step}/{TOTAL_TIMESTEPS} steps ({progress_pct:.1f}%)"
            agent.writer.add_text("Training/Status", status_msg, global_step=step)

        ep_reward += reward
        ep_steer_diff += info.get("steer_diff", 0.0)
        ep_wp_dev += info.get("wp_dev", 0.0)

        #Training Agent
        warmup_gate = (step >= UPD_START_STEP)
        if warmup_gate and step % UPD_EVERY == 0:
          
            for _ in range(G): 
                agent.train(step)

        # Handle Episode End
        if done:
            #Episode finished
            train_ep_len = max(env.step_counter, 1)
            agent.replay_buffer.commit_episode()

            logger.store(EpRet=float(ep_reward),
                         EpLen=float(train_ep_len),
                         SteerJerk=float(ep_steer_diff / train_ep_len))

            if step - last_eval >= EVAL_EVERY and (step >= UPD_START_STEP):
                last_eval = step
                agent.actor.eval(); agent.critic.eval()
                
                agg = evaluate_policy(agent, env, c_proxy.Env["num_actions"], agent.device,
                                      gamma = agent.gamma, n_episodes = 5,
                                      hidden_size=agent.hidden_size,
                                      max_steps = SECONDS_PER_EPISODE * 20)
                
                agent.actor.train(); agent.critic.train()
                log_eval(agent, agg, step)

                logger.log_tabular('Epoch/Eval', episodes)
                logger.log_tabular('TotalEnvInteracts', step)
                logger.log_tabular('EpRet',with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('SteerJerk', average_only=True)
                
                # Add eval metrics to the table
                logger.log_tabular('Eval_SuccessRate', agg["success"])
                logger.log_tabular('Eval_MCTE', agg["MCTE"])
                logger.log_tabular('Q_MC_Bias', agg["Q_minus_MC_bias"])
                
                # This prints the clean ASCII table and flushes the stored data!
                logger.dump_tabular()

                score = (agg["success"], agg["ret_scaled"])  # SR primary
                if score > best_score:
                    best_score = score
                    th.save({'actor': agent.actor.state_dict(),
                             'critic': agent.critic.state_dict()},
                             os.path.join(modelsdir, "best_model.pth"))
            episodes += 1

            global_step_current = step

            agent.writer.add_scalar("Env/Episode_Reward", ep_reward, global_step_current)
            agent.writer.add_scalar("Diag/Net_Displacement", info["distance"], global_step_current)
            agent.writer.add_scalar("Env/Distance_Travelled", info.get("odometer", info["distance"]), global_step_current)
            agent.writer.add_scalar("Env/Episode_Length", train_ep_len, global_step_current)
            
            # Averages over the episode
            agent.writer.add_scalar("Comfort/Avg_Steering_Jerk", ep_steer_diff / train_ep_len, global_step_current)
            agent.writer.add_scalar("Comfort/Avg_Waypoint_Dev", ep_wp_dev / train_ep_len, global_step_current)
            
            # Log termination reasons as binary events (1 if true, 0 if false)
            term = info.get("termination_reason", {})
            agent.writer.add_scalar("Safety/Collision", int(term.get("collision", False)), global_step_current)
            agent.writer.add_scalar("Safety/Fatal_Lane", int(term.get("fatal_lane", False)), global_step_current)
            agent.writer.add_scalar("Safety/Red_Light", int(term.get("red_light", False)), global_step_current)

            #Reset
            ep_reward = 0.0
            ep_steer_diff = 0.0
            ep_wp_dev = 0.0

            obs, _ = env.reset()
                        
            # Reset Inference Hidden State for the new episode
            inference_hidden_state = (th.zeros(1, 1, 256).to(agent.device), 
                                      th.zeros(1, 1, 256).to(agent.device))

            prev_a = np.zeros((1, 1, c_proxy.Env["num_actions"]), dtype=np.float32)
                
    # Final Save
    th.save({'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict()},
            os.path.join(modelsdir, f"final_seed{SEED}.pth"))

    env.close()
    print(' Training complete, models saved.')
    kill_carla(PORT)