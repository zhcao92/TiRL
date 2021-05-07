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

from TestRule import IDM

EPISODES=100

class TiRLAgent(object):

    def __init__(self, env):

        self.env = env

        load_path_rl="/home/icv/Trustworth/TiRL/models/sac-5"
        log_path_rl="/home/icv/Trustworth/TiRL/data/sac-5"
        
        self.model_rl = SAC.load(load_path_rl, env=env, tensorboard_log=log_path_rl)

        load_path_rule="/home/icv/Trustworth/TiRL/models/sac_rule3"
        log_path_rule="/home/icv/Trustworth/TiRL/data/sac_rule3"
        
        self.model_rule = SAC.load(load_path_rule, env=env, tensorboard_log=log_path_rule)
        self.agent_rule = IDM(env)

        print("load model successfully")

        self.reset()
    
    def reset(self):
        self.rule_step = 0
        self.rl_step = 0

    def act(self, state):

        action_rl, _states = self.model_rl.predict(state,deterministic=True)
        action_rule = self.agent_rule.act(state)

        q_rl = self.model_rl.get_q_value([state],np.array([action_rl]))
        q_rule = self.model_rule.get_q_value([state],np.array([action_rule])[None])

        # print("----",float(q_rule),float(q_rl))
        # action = (action_rl, action_rule)[float(q_rule)>float(q_rl)]

        if q_rule > q_rl: 
            action = action_rule
            self.rule_step = self.rule_step+1
        else:
            action = action_rl
            self.rl_step = self.rl_step+1

        return action

if __name__ == '__main__':

    # Load the model
    # model = load_model(MODEL_PATH)

    # Create environment
    env = CarEnv(random_env=False)

    # Create Agent
    agent = TiRLAgent(env)

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

        print("Episode Reward:",episode_reward," RL Step: ",agent.rl_step," Rule Step: ",agent.rule_step)

        with open("data/TiRL/RL_results.txt", "a") as result_recorder:
            result_recorder.write(str(episode_reward)+' '+str(agent.rl_step)+''+str(agent.rule_step)+'/n')

        agent.reset()

        
