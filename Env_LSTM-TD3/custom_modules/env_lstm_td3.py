
import glob
import os 
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import socket
import sys
import random
import time
import math
import webbrowser 
import numpy as np 
import cv2        
import gymnasium as gym
from gymnasium import spaces                              
import carla
import subprocess
from collections import deque

try: 
    sys.path.append(glob.glob('../carla/carla/dist/carla-*%d.%d-%s.egg'%(
      sys.version_info.major,
      sys.version_info.minor,
      'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SHOW_PREVIEW = os.environ.get("SHOW_PREVIEW", "0") == "1"
NO_RENDERING = False
N_CHANNELS = 3

IM_WIDTH = 288
IM_HEIGHT = 120

SECONDS_PER_EPISODE = 35
SYNCHRONOUS_MODE = True
SPIN = 10
DISCOUNT = 0.99

class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4

    def __init__(self, seed=0, port=2010):
        super(CarEnv, self).__init__()
        self.env_seed = seed
        self.width = IM_WIDTH
        self.height = IM_HEIGHT
        self.image_for_CNN = None
        self.actor_list = []
        self.front_camera = None
        self.collision_hist = []
        print(f"Connecting to CARLA server on port {port}...")
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(30.0)
       
        self.world = self.client.get_world()
        current_map = self.world.get_map().name
        target_map = 'Town10HD_Opt'
        
        if current_map != target_map:
            logger.info(f"Loading map {target_map} to allow layer toggling...")
            self.world = self.client.load_world(target_map)     
        try:
            self.world.unload_map_layer(carla.MapLayer.Buildings)
            self.world.unload_map_layer(carla.MapLayer.Particles)  
            logger.info("Successfully unloaded Buildings and Particles layers for improved performance.")
        except Exception as e:
            logger.warning(f"Could not unload map layers: {e}")
        
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.world.tick()
        self.settings = self.world.get_settings()

        if SYNCHRONOUS_MODE:
            self.settings.fixed_delta_seconds = 0.05
        self.settings.no_rendering_mode = NO_RENDERING
        self.settings.synchronous_mode = SYNCHRONOUS_MODE
        self.world.apply_settings(self.settings)
        self.steering_lock = False
        self.steering_locked_start = None   
        self.prev_steer = 0.0
        self.image_for_CNN = None
        self.prev_location = None                        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(N_CHANNELS, self.height, self.width), dtype=np.uint8),
            "kinematics": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })
        self.spectator = None
        if not NO_RENDERING:
            self.spectator = self.world.get_spectator()
        
    def clean_up(self):
        for actor in self.actor_list:
            if actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.error(f"Failed to destroy actor {actor.id}: {e}")
        self.world.tick()

    def reset(self,seed=None):
        super().reset(seed=seed)
        self.clean_up()
        self.step_counter = 0
        self.total_distance_travelled = 0.0
        self.stationary_steps = 0
        self.collision_hist = []
        
        self.lane_invasion_hist = []
        self.actor_list = []
        self.prev_phi = 0.0
        self.steering_lock = False
        self.steering_lock_steps = 0
        self.prev_location = None
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.front_camera = None
        self.world.tick()
        self.world = self.client.get_world()  # Refresh the world reference after cleanup
        self.map = self.world.get_map()
        Weather_presets =[carla.WeatherParameters.ClearNoon,
                          carla.WeatherParameters.HardRainNoon,
                          carla.WeatherParameters.WetCloudySunset,
                          carla.WeatherParameters.MidRainyNoon,
                          carla.WeatherParameters.WetSunset]
        self.world.set_weather(random.choice(Weather_presets))
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        angle_adj = random.randrange(-SPIN, SPIN, 1)
        self.transform.rotation.yaw += angle_adj

        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"90")

        transform = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        for _ in range(10):
            self.world.tick()
        
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_invasion_data(event))

        while self.front_camera is None:
            self.world.tick()
            
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_locked_start = None
        self.step_counter = 0

        single_frame = np.transpose(self.front_camera,(2,0,1))
        self.image_for_CNN = single_frame
        obs = {
            "image": self.image_for_CNN,
            "kinematics": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        self.next_checkpoint = 50.0
        return obs, {}
                                                                                          
    def step(self, action):

        self.lane_invasion_hist.clear()
        # Pre-tick : state s_t for progress and control decision
        trans_before = self.vehicle.get_transform()
        loc_before = trans_before.location
        v_pre = self.vehicle.get_velocity()
        kmh_pre = int(3.6* math.sqrt(v_pre.x**2 + v_pre.y**2 + v_pre.z**2))
   
        if self.SHOW_CAM and self.spectator is not None:
            fv = trans_before.get_forward_vector()
            cam_loc = trans_before.location - carla.Location(x=fv.x*6, y=fv.y*6, z=0.0) + carla.Location(z=3.5)
            cam_rot = carla.Rotation(pitch=-15.0, yaw=trans_before.rotation.yaw, roll=0.0)
            self.spectator.set_transform(carla.Transform(cam_loc, cam_rot))

        self.step_counter += 1
        
        # Action - control (steer attenuation uses pre-tick speed)
        old_steer = self.prev_steer
        raw_steer = float(action[0])
        raw_throttle_brake = float(action[1]) 
        speed_factor = min(kmh_pre/40.0, 1.0)
        max_steer_scale = 1.0 - (0.4*speed_factor)
        attenuated_steer = raw_steer*max_steer_scale

        #Proportional Steering to avoid jerk       
        STEER_ALPHA = 0.45
        steer = (STEER_ALPHA*attenuated_steer) + ((1.0 -STEER_ALPHA)*old_steer)
        THROTTLE_ALPHA = 0.30
        smoothed_throttle_brake = (THROTTLE_ALPHA * raw_throttle_brake) + ((1.0 - THROTTLE_ALPHA)*self.prev_throttle)
        self.prev_throttle = smoothed_throttle_brake

        if smoothed_throttle_brake>=0.0:
            throttle, brake = float(smoothed_throttle_brake), 0.0
        else:
            throttle, brake = 0.0, float(abs(smoothed_throttle_brake))
        
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        
        # Advance physics
        if SYNCHRONOUS_MODE:
            self.world.tick()
        
        # Post-tick state s_t+1 for rew and obs
        trans = self.vehicle.get_transform()
        loc_after = trans.location
        step_distance = loc_before.distance(loc_after)
        self.total_distance_travelled += step_distance
        self.prev_location = loc_after
        distance_travelled = self.total_distance_travelled
        net_disp = self.initial_location.distance(loc_after)
       
        v = self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        red_light = False
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red:
                red_light = True
        if kmh < 1.0 and not red_light:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0
        stationary_limit = 100
       
        waypoint = self.map.get_waypoint(loc_after, project_to_road=True, lane_type=carla.LaneType.Driving)
        wp_dev = waypoint.transform.location.distance(loc_after)
        v_fwd = trans.get_forward_vector()
        w_fwd = waypoint.transform.get_forward_vector()
        heading_alignment = (v_fwd.x * w_fwd.x + v_fwd.y * w_fwd.y)

        cross_hv = (w_fwd.x*v_fwd.y) - (w_fwd.y * v_fwd.x)
        heading_err = math.atan2(cross_hv, heading_alignment)
        norm_heading_err = float(np.clip(heading_err / math.pi, -1.0, 1.0))

        # lane centering sides via signed deviations
        dx = loc_after.x - waypoint.transform.location.x
        dy = loc_after.y - waypoint.transform.location.y
        cross_z = (w_fwd.x *dy) - (w_fwd.y*dx)
        signed_wp_dev = math.copysign(wp_dev, cross_z) if cross_z != 0.0 else wp_dev
        norm_signed_dev = float(np.clip(signed_wp_dev / 3.0, -1.0, 1.0))

        terminated = False
        truncated = False
      
        if abs(steer)>0.9:
            self.steering_lock_steps += 1
        else:
            self.steering_lock_steps = 0
        lock_duration = self.steering_lock_steps*0.05
         
        collision = len(self.collision_hist) != 0
        fatal_lane_invasion = False
        soft_lane_invasion = False

        for marking_type in self.lane_invasion_hist:
            if marking_type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid, 
                                carla.LaneMarkingType.SolidBroken, carla.LaneMarkingType.BrokenSolid]:
                fatal_lane_invasion=True
            elif marking_type == carla.LaneMarkingType.Broken:
                soft_lane_invasion = True

        signal_jump = red_light and (kmh > 2.0)
        lock_exceeded = lock_duration > 3.0
        wp_dev_exceeded = (wp_dev > 3.5 and self.step_counter > 40)

        if collision or fatal_lane_invasion or lock_exceeded or signal_jump:
            terminated = True
            reward = -1.0
            logger.debug(f"Terminated at step{self.step_counter}. Collision={collision},"
                        f"FatalLane={fatal_lane_invasion}, Lock={lock_exceeded}, Signalskip={signal_jump}")
        elif wp_dev_exceeded:
            terminated = True
            reward = -1.0
            logger.debug(f"Terminated at step{self.step_counter}. wp_dev={wp_dev:.2f} exceeded 3.5 ")

        elif self.stationary_steps >= stationary_limit and self.step_counter > 40:
            terminated = True
            reward =-1.0

        else:
            reward = self.calculate_reward(v, w_fwd, net_disp, steer, old_steer,
                                            wp_dev, heading_alignment, red_light)
            if soft_lane_invasion:
                reward -= 0.25

        MAX_STEPS = SECONDS_PER_EPISODE * 20
        if (not terminated) and self.step_counter >= MAX_STEPS:
            truncated = True
            reward = float(np.clip(reward +0.5, -1.0, 1.0))

        if self.step_counter % 1000 == 0:
            logger.debug(f"Act:{steer:0.2f}, Raw_TB:{raw_throttle_brake:0.2f}, kmph:{kmh:0.2f},"
                        f"wp_dev:{wp_dev:0.2f}, Heading:{heading_alignment:0.2f}, reward:{reward:0.2f}")

        # next observation : single post-tick frame and post tick kinematics
        if self.front_camera is not None:
            self.image_for_CNN = np.transpose(self.front_camera, (2, 0, 1))
            single_image = self.image_for_CNN
            if self.SHOW_CAM:
                cv2.imshow("CARLA Driver Preview", self.front_camera); cv2.waitKey(1)
        else:
            single_image = np.zeros((N_CHANNELS, self.height, self.width), dtype=np.uint8)
        
        norm_speed = np.clip(float(kmh)/90.0, 0.0, 1.0)
        red_light_val = 1.0 if red_light else 0.0
        
        obs = {
            "image": single_image,
            "kinematics": np.array([norm_speed, float(steer), float(old_steer), 
                                    norm_signed_dev, float(red_light_val),
                                     float(self.prev_throttle),
                                      norm_heading_err], dtype=np.float32) }
        
        info = {"Kmph": float(kmh), "wp_dev":float(wp_dev), 
                "steer_diff": float(abs(steer-old_steer)), 
                "distance": float(net_disp),
                "odometer" : float(distance_travelled),
                "termination_reason": {"collision":collision, "fatal_lane":fatal_lane_invasion, 
                                       "soft_lane": soft_lane_invasion, "lock": lock_exceeded,
                                       "red_light": signal_jump, "wp_dev_exceeded": wp_dev_exceeded,
                                       "timeout": truncated}}
        
        self.prev_steer = steer
        
        return obs, reward, terminated, truncated, info

    def lane_invasion_data(self, event):
        for marking in event.crossed_lane_markings:
            self.lane_invasion_hist.append(marking.type)

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):

            i = np.array(image.raw_data, dtype=np.uint8)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            self.front_camera = i3

    def calculate_reward(self, v, w_fwd, net_disp, current_steer, 
                         prev_steer, wp_dev, heading_alignment, red_light):
        
        forward_speed = (v.x * w_fwd.x) + (v.y * w_fwd.y)
    #smooth centering
        c_f = 1.0
        center_factor = math.exp(-(wp_dev / c_f)**2)    # 1.0 at lane center

    #heading alignment
        align = max(0.0, heading_alignment)

    #Speed smoothing
        v_target, s_v = 8.0, 4.0
        speed_r = math.exp(-((forward_speed - v_target)/ s_v)**2)

        if red_light:
            r_main = math.exp(-(forward_speed / 1.0) **2)
        else:   
            r_main = center_factor * align*speed_r

    # Potential based progress
        PHI_SCALE = 50.0
        phi_now = math.tanh(net_disp / PHI_SCALE)
        F_prog = (DISCOUNT*phi_now)- self.prev_phi
        self.prev_phi = phi_now
        W_PROG = 0.5

        steer_diff = abs(current_steer - prev_steer)
        comfort = 0.1 * min(steer_diff, 1.0)

        reward = r_main + W_PROG * F_prog - comfort

        return float(np.clip(reward, -1.0, 1.0))

    def close(self):
        self.clean_up()
        self.world.tick()
        logger.info("Environment closed and cleaned up.")

        if hasattr(self, 'spectator') and self.spectator is not None:
            self.spectator = None