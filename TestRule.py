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

from TestScenario import CarEnv

EPISODES=100

class IDM(object):

    def __init__(self, env):

        self.T = 1.5
        self.g0 = 4
        self.a = 0.73 
        self.b = 1.67
        self.delta = 4
        self.length_x = 5 # front vehicle length
        self.env = env

    def act(self, state):

        ori_state = state * (self.env.observation_space.high - self.env.observation_space.low)
        ego_x = ori_state[0]
        ego_v = ori_state[1]
        lead_x = ori_state[2]
        lead_v = ori_state[3]

        v0 = 100 / 3.6  # speed limit

        if v0 <= 0.0:
            return 0.0
        
        a = self.a
        b = self.b
        g0 = self.g0
        T = self.T
        delta = self.delta
        
        dv = ego_v - lead_v

        g = np.linalg.norm(lead_x - ego_x) - self.length_x

        # adaptive following speed, need to be checked in real car
        if g <= 0:
            return 0

        g1 = g0 + T * ego_v + ego_v * dv / (2 * np.sqrt(a * b))

        acc = a * (1 - pow(ego_v/v0, delta) - (g1/g) * ((g1/g)))

        return acc

if __name__ == '__main__':
    
    # Load the model
    # model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv(random_env=False)

    # Create Agent
    agent = IDM(env)

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Loop over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        
        print('Restarting episode')

        # Reset environment and get initial state
        current_state = env.reset()
        env.collision_hist = []

        episode_reward = 0
        step = 1

        done = False

        # Loop over steps
        while True:

            # For FPS counter
            step_start = time.time()
            action = agent.act(current_state)
            new_state, reward, done, _ = env.step(action)            
            episode_reward += reward

            # Set current step for next loop iteration
            current_state = new_state
            step += 1

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)

        print("Episode Reward:",episode_reward)
        
        # Record Results DATA
        with open("data/TiRL/rule_results.txt", "a") as result_recorder:
            result_recorder.write(str(episode_reward).join('/n'))

