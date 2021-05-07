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

from stable_baselines import SAC

from TestScenario import CarEnv

EPISODES=100

class RLAgent(object):

    def __init__(self, env):

        self.T = 1.5
        self.g0 = 4
        self.a = 0.73 
        self.b = 1.67
        self.delta = 4
        self.decision_dt = 0.75
        self.length_x = 5 # front vehicle length
        self.env = env

        load_path="/home/icv/Trustworth/TiRL/models/sac-5"
        log_path="/home/icv/Trustworth/TiRL/data/sac-5"
        
        self.model = SAC.load(load_path,env=env, tensorboard_log=log_path)
        print("load model successfully")

    def act(self, state):

        action, _states = self.model.predict(state,deterministic=True)   

        return action

if __name__ == '__main__':

    # Load the model
    # model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv(random_env=False)

    # Create Agent
    agent = RLAgent(env)

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
        with open("data/TiRL/RL_results.txt", "a") as result_recorder:
            result_recorder.write(str(episode_reward).join('/n'))