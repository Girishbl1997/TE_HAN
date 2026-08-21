import math
import numpy as np
import carla

class L1Reward:
    W_PROG=0.5
    W_BRAKE=1.0
    W_OFF=0.1
    W_STOP=0.25
    ZONE_M, DISCOUNT = 25.0, 0.99

    def __init__(self):
        self._prev_phi = 0.0 

    def reset(self):
        self._prev_phi = 0.0

    def step(self, env, offset_cmd, gate_cmd):
        v = env.vehicle.get_velocity()
        wp = env.map.get_waypoint(env.vehicle.get_location(), project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
        wf = wp.transform.get_forward_vector()
        fs = (v.x*wf.x) + (v.y*wf.y)

        phi = math.tanh(env.initial_location.distance(env.vehicle.get_location())/50.0)
        F_prog = self.DISCOUNT*phi - self._prev_phi 
        self._prev_phi = phi

        dist_norm, onehot = env._next_tl_info(max_d=40.0)
        in_zone_red = (onehot[0] >= 0.5) and (dist_norm*40.0 < self.ZONE_M)
        r = self.W_PROG * F_prog

    # Anticipatory Brake
        if in_zone_red:
            r += self.W_BRAKE*math.exp(-fs**2)

    #needless stop and offset
        if gate_cmd >= 0.5 and not in_zone_red:
            r -= self.W_STOP
        r -= self.W_OFF * abs(offset_cmd)

        return float(np.clip(r, -1.0, 1.0))