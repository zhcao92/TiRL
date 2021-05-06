import gym
import numpy as np
import gym_routing


# from stable_baselines.ddpg.policies import MlpPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines import DDPG
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC

from TestRule import IDM

from TestScenario import CarEnv

log_path="/home/icv/Trustworth/stable-baselines/data/sac"
load_path="/home/icv/Trustworth/stable-baselines/models/sac"
save_path="/home/icv/Trustworth/stable-baselines/models/sac"



env = CarEnv(random_env = False)

# the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))


try:
    model = SAC.load(load_path,env=env, tensorboard_log=log_path)
    # model = DDPG.load(load_path,env=env, tensorboard_log=log_path)
    print("SAC: Load saved model")

except:
    # model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, gamma=0.99, nb_train_steps=20, nb_rollout_steps=100, nb_eval_steps=50, tensorboard_log=log_path,full_tensorboard_log=True, save_path=save_path)
    model = SAC(MlpPolicy, env, verbose=1)

    print("SAC: Build new model")

model.learn(total_timesteps=100000, rule_based_training=False)
model.save(save_path)

del model # remove to demonstrate saving and loading

model = SAC.load(load_path)
print("Load model to test")
obs = env.reset()


while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()
