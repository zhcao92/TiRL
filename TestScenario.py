import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass
try:
	sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np

import math
import random
from collections import deque
import cv2
from tqdm import tqdm

# For rl
import gym
from gym import core, error, spaces, utils
from gym.utils import seeding

MAP_NAME = 'Town01'
LANE_YAW = 0
LANE_DIR = 0
EPISODES=100
SECONDS_PER_EPISODE=60

class CarEnv:

    def __init__(self, random_env = False):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name != MAP_NAME:
            self.world = self.client.load_world(MAP_NAME)
        self.world.set_weather(carla.WeatherParameters(cloudiness=0, precipitation=30.0, sun_altitude_angle=70.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.mercedes-benz.coupe'))
        self.vehicle_bpa = random.choice(self.blueprint_library.filter('vehicle.bmw.grandtourer'))

        self.lane_dir = LANE_DIR
        self.collision_hist = []
        self.last_acc = 0
        self.range_initial_leader = 25
        self.v_initial_leader = 8

        # RL settings
        # Set action space
        low_action = np.array([-1]) 
        high_action = np.array([1])  
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        self.state_dimension = 4
        # state = np.array([ego_x, ego_v, lead_x, lead_v])
        low  = np.array([120, 0, 120, 0])
        high = np.array([400, 30, 400, 30])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # vehicles init
        self.vehicle = None 
        self.va_leader = None 

        # case num
        self.case_num = 0

        self.random_env = random_env

    def get_case_parameters(self):

        if not self.random_env:
            self.case_num = self.case_num + 1
            self.range_initial_leader = 40 - 0.25*self.case_num
            self.v_initial_leader = 12 - 0.08*self.case_num
        else:
            self.range_initial_leader = random.uniform(20,40)
            self.v_initial_leader = random.uniform(5,12)

    def reset(self):
        
        # check environments
        self.episode_start = time.time()      

        if self.va_leader is not None:
            self.va_leader.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        self.world.tick()
        self.get_case_parameters()

        # Ego vehicle state setting
        ego_vehicle_init_location = 320
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, carla.Transform(
                                                    carla.Location(ego_vehicle_init_location,129.750,0.3),
                                                    carla.Rotation(0,180,0))
                                        )

        self.vehicle.set_target_velocity(carla.Vector3D(x=-10,y=0,z=0))

        # Lead vehicle state setting
        lead_vehicle_init_location = ego_vehicle_init_location - self.range_initial_leader

        self.va_leader = self.world.spawn_actor(self.vehicle_bpa, carla.Transform(
                                        carla.Location(lead_vehicle_init_location,129.750,0.3),
                                        carla.Rotation(0,180,0)
                                        ))
        self.va_leader.set_target_velocity(carla.Vector3D(x=-self.v_initial_leader,y=0,z=0))
        self.world.tick()

        # Get new state
        ego_x = self.vehicle.get_location().x
        ego_v = abs(self.vehicle.get_velocity().x)

        lead_x = self.va_leader.get_location().x
        lead_v= abs(self.va_leader.get_velocity().x)

        state = np.array([ego_x, ego_v, lead_x, lead_v])
        
        return (state-self.observation_space.low)/(self.observation_space.high-self.observation_space.low)

        
    def step(self, action):

        self.world.tick()

        # control ego vehicle
        throttle = max(0,float(action))
        brake = max(0,-float(action))
        self.vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake))

        ego_x = self.vehicle.get_location().x
        ego_v = abs(self.vehicle.get_velocity().x)

        lead_x = self.va_leader.get_location().x
        lead_v= abs(self.va_leader.get_velocity().x)

        # control lead vehicle
        if abs(ego_x-lead_x) <= self.range_initial_leader:
            self.va_leader.apply_control(carla.VehicleControl(throttle = 0.0, brake = 0.3))
            if lead_v <= 2:
                self.va_leader.apply_control(carla.VehicleControl(throttle = 0.3, brake = 0.0))
        else:
            self.va_leader.set_target_velocity(carla.Vector3D(x=-self.v_initial_leader,y=0,z=0))

        # get reward
        reward = ego_v / 10
        state = np.array([ego_x, ego_v, lead_x, lead_v])

        done = False
        if abs(ego_x - lead_x) < 6: # before we have a collision sensor..
            reward = -100
            print('----collision')
            done = True

        if lead_x <= 150:
            print('----finshed')
            done = True

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return (state-self.observation_space.low)/(self.observation_space.high-self.observation_space.low), reward, done, None
