import gymnasium as gym

class DotDict(dict):
    """
    A dictionary that supports dot notation for attribute access.
    Converts nested dictionaries recursively so BaseAgent can access c.Env.state_type
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item):
        if item in self:
            return self[item]
        return None

    def __setattr__(self, key, value):
        self[key] = value

class AgentConfigFacade:
    """
    A robust Proxy/Facade that flattens the nested ConfigFile object and
    wraps dictionaries in DotDicts to support strict attribute access.
    """
    def __init__(self, c, env: gym.Env = None):
        self._c = c
        self._flat_config = {}

        env_dict = getattr(c, "Env", {})
        agent_dict = getattr(c, "Agent", {})
        train_dict = getattr(c, "Train", {})

        for d in [env_dict, agent_dict, train_dict]:
            if isinstance(d, dict):
                self._flat_config.update(d)

        if env is not None:
            self._flat_config["action_high"] = float(env.action_space.high[0])
            self._flat_config["action_low"]  = float(env.action_space.low[0])
        else:
            self._flat_config["action_high"] = 1.0
            self._flat_config["action_low"]  = -1.0

        if "sequence_length" in self._flat_config:
            self._flat_config["history_length"] = self._flat_config["sequence_length"]
        if "actor_lr" in self._flat_config:
            self._flat_config["lr_actor"] = self._flat_config["actor_lr"]
        if "critic_lr" in self._flat_config:
            self._flat_config["lr_critic"] = self._flat_config["critic_lr"]

        if self._flat_config.get("loss") == "MSE":
            self._flat_config["loss"] = "MSELoss"
        elif self._flat_config.get("loss") in ["SmoothL1", "Huber"]:
            self._flat_config["loss"] = "SmoothL1Loss"

        self._flat_config.setdefault("grad_rescale", False)
        self._flat_config.setdefault("grad_clip", False)
        self._flat_config ["buffer_transitions"] = int(self._flat_config.get("buffer_length", 12000))
        # --- CRITICAL FIX FOR LSTMDDPG Constructor ---
        # The agent logic expects a nested dict for recurrent parameters
        agent_name = agent_dict.get("agent_name", "LSTMTD3DictAgent")
        if agent_name not in agent_dict:
            agent_dict[agent_name] = {}
            
        agent_dict[agent_name]["target_noise"] = agent_dict.get("policy_noise", 0.2)
        agent_dict[agent_name]["target_noise_clip"] = agent_dict.get("policy_noise_clip", 0.5)
        agent_dict[agent_name]["pol_upd_delay"] = agent_dict.get("policy_delay", 2)
        # Bypasses the exact line 29 KeyError you just hit:
        agent_dict[agent_name]["history_length"] = agent_dict.get("sequence_length", 8)
        agent_dict[agent_name]["use_past_actions"] = agent_dict.get("use_past_actions", False)

        self.Env = DotDict(env_dict)
        self.Agent = DotDict(agent_dict)
        self.Train = DotDict(train_dict)
        # Dummy values to skip the checks, later custom agent code will overwrite.
        self._flat_config["state_shape"] = 1 
        self._flat_config["state_type"] = "feature"
        self._flat_config["buffer_length"] = 1
        self.Env.state_shape = 1
        self.Env.state_type = "feature"
        self.Agent.buffer_length = 1
        
        if getattr(self.Env, "image_shape", None) is None:
            self.Env.image_shape = [3, 120, 288]
        if getattr(self.Env, "kin_shape", None) is None:
            self.Env.kin_shape = [7]

    def __getattr__(self, item):
        if item in self._flat_config:
            return self._flat_config[item]
        if hasattr(self._c, item):
            return getattr(self._c, item)
        return None
