
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from tud_rl.agents._continuous.LSTMDDPG import LSTMDDPGAgent
from tud_rl.common.configparser import ConfigFile
from custom_modules.nets_dict import LSTMDict_Actor, Double_LSTMDict_Critic
from custom_modules.recurrent_dict_buffer import RecurrentDictReplayBuffer

class LSTMTD3DictAgent(LSTMDDPGAgent):
    def __init__(self, c: ConfigFile, agent_name):
        super().__init__(c, agent_name, init_critic=False)
        self.state_type = "dict"
        self.state_shape = {"image": tuple(c.Env["image_shape"]),
                            "kinematics": tuple(c.Env["kin_shape"])}
        self.hidden_size = 256
        self.burn_in = 8  # frozen fairness variable B

        self.actor = LSTMDict_Actor(self.num_actions, self.hidden_size).to(self.device)            
        self.critic = Double_LSTMDict_Critic( self.num_actions, self.state_shape,
                                              self.hidden_size).to(self.device)
        
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
       
        agent_cfg = c.Agent[agent_name]

    # attributes and hyperparameters
        self.tgt_noise      = agent_cfg["target_noise"]
        self.tgt_noise_clip = agent_cfg["target_noise_clip"]    
        self.pol_upd_delay  = agent_cfg["pol_upd_delay"]
        
    # counter for policy update delay
        self.pol_upd_cnt = 0
        self.grad_steps = 0
        self.CRITIC_WARMUP = 2000
        
    # freeze target critic nets with respect to optimizers to avoid unnecessary computations
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr =self.lr_critic)
    # since, the base version of LSTMTD3 uses LSTMDDPG as a base, actor optimizer is already defined, but we need to redefine
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr =self.lr_actor)
        buf_cap = int(c.buffer_transitions)
        self.replay_buffer = RecurrentDictReplayBuffer(state_shape=self.state_shape,
                                                       action_dim=self.num_actions,
                                                       max_size=buf_cap,
                                                       max_episode_length= int(c.Env["max_episode_steps"]))

    def train(self, global_env_step):
        """Samples from RecurrentDictReplayBuffer, updates with masked BPTT and Causal Isolation"""        
    # sample from custom buffer
        self.train_update_step = global_env_step
        s, a, r, s2, d, m = self.replay_buffer.sample(self.batch_size, self.history_length, self.burn_in)
        m = m.unsqueeze(-1)

        a_past = torch.cat([torch.zeros(self.batch_size, 1, self.num_actions).to(self.device), a[:, :-1, :]], dim = 1)
        a2_past = a 

        hidden_a = (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                    torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))
        hidden_c1 = (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                     torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))
        hidden_c2 = (torch.zeros(1, self.batch_size, self.hidden_size).to(self.device),
                     torch.zeros(1, self.batch_size, self.hidden_size).to(self.device))
    # Increment first
        self.grad_steps += 1      

    #-------- Train CRITIC --------       
        # clear gradients
        self.critic_optimizer.zero_grad()
        # online encoder
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f_s = self.critic.encode(s)   
        # Q-estimates
            Q1, Q2, _, _ = self.critic(f_s, a, hidden_c1, hidden_c2, a_past) 
            with torch.no_grad():
                # Target encoder
                f_s2 = self.target_critic.encode(s2)
                target_a, _, _ = self.target_actor(f_s2, hidden_a, a2_past)
                noise = torch.clamp(torch.randn_like(target_a) * self.tgt_noise,
                                    -self.tgt_noise_clip, self.tgt_noise_clip)
                target_a = torch.clamp(target_a + noise, -1.0, 1.0)
        # compute target Q-values
                q1_target, q2_target, _, _ = self.target_critic(f_s2, target_a, hidden_c1, hidden_c2, a2_past)
                q_target = torch.min(q1_target, q2_target)

                y = r + self.gamma*(1.0 - d)*q_target 
            q1_loss = (((Q1 - y)**2)*m).sum()/m.sum()
            q2_loss = (((Q2 - y)**2)*m).sum()/m.sum()
            critic_loss = q1_loss + q2_loss        
    # compute gradients
        critic_loss.backward()
    # gradient clipping
        gn_c = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 1.0) # pre-clip measures
        if self.grad_steps % 50 == 0:
            self.writer.add_scalar("Diag/GradNorm_critic_preclip", gn_c.item(), self.train_update_step)
    # perform optimizing step
        self.critic_optimizer.step()
    # log critic training
        if self.grad_steps % 50 == 0:
            with torch.no_grad():
                q_div = ((Q1 - Q2).abs() *m).sum() / m.sum()
                self.writer.add_scalar("Diag/Q1_minus_Q2_absmean", q_div.item(), self.train_update_step)
                 
    #-------- Train ACTOR --------                      
        do_delayed = (self.grad_steps % self.pol_upd_delay == 0)
        if do_delayed and self.grad_steps > self.CRITIC_WARMUP:
    # freeze critic so no gradient computations are wasted while training actor
            for param in self.critic.parameters():
                param.requires_grad = False
    # clear gradients
            self.actor_optimizer.zero_grad()
            f_s_actor = f_s.detach()   # stop-grad to encoder
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                curr_a, raw_logits, _ = self.actor(f_s_actor, hidden_a, a_past)  # using actor's extractor
                q1_curr, _ = self.critic.single_forward(f_s_actor, curr_a, hidden_c1, a_past)
                
                logit_loss = 1e-3 * (raw_logits**2 * m).sum() / torch.clamp(m.sum(), min = 1.0)
                policy_loss = -(q1_curr*m).sum()/torch.clamp(m.sum(), min = 1.0)
                actor_loss = policy_loss + logit_loss
        # compute gradients
            actor_loss.backward()        
    # gradient clipping
            gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
   # perform step with optimizer
            self.actor_optimizer.step()
    # unfreeze critic so it can be trained in next iteration
            for param in self.critic.parameters(): param.requires_grad = True     
            if self.grad_steps % 50 == 0:
    # log actor training
                self.writer.add_scalar("Diag/GradNorm_Actor_preclip", gn_a.item(), self.train_update_step)
                with torch.no_grad():
                    sat = ((curr_a.abs() > 0.99).float() *m).sum() / (m.sum() * self.num_actions)       
                    self.writer.add_scalar("Diag/Action_Saturation", sat.item(), self.train_update_step)
                self.writer.add_scalar("Loss/Actor", actor_loss.detach().item(), self.train_update_step)               
            self.pol_upd_cnt += 1
        
        if do_delayed:
            self.polyak_update()
    


   
