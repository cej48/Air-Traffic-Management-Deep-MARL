# Christopher Jones 2023, adapted from cleanRL

import argparse
import os
import random
import time
from distutils.util import strtobool

#  import gym
import sys
sys.path.append("build/")
import PyATMSim
from random import randint
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

import numpy as np
# import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

load_from_file = False


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e5),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    # parser.add_argument("--batch-size", type=int, default=512,
    #     help="the batch size of sample from the reply memory")
    parser.add_argument("--batch-size", type=int, default=512,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.2,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 1024)

        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc_mu(x))
        
        return x * self.action_scale + self.action_bias


class EnvWrap():
    def __init__(self):
        self.env = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0, 25)
        self.env.step()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape = (3,))
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape  = np.array(self.env.traffic[0].get_observation()).shape)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
    def reset(self):
        return self.env.reset()
    def step(self):
        return self.env.step()

if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = EnvWrap()
    actor = Actor(envs)
    # device = "cpu"
    if device == "cuda":
        actor.load_state_dict(torch.load("./saved_models/good/actor.pt"))
        actor.to(device)
    else:
        actor.load_state_dict(torch.load("./saved_models/good/actor.pt", map_location=device))
    envs.single_observation_space.dtype = np.float32
    obs = envs.reset()

    states = {i : i.get_observation() for i in envs.env.traffic}  
    actions = {i : [] for i in envs.env.traffic}
    # for each step
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here

        states = {i : i.get_observation() for i in envs.env.traffic}  
        actions={}

        for traffic in states:
            reward=-130 + traffic.reward
            if reward>0:
                print(reward)
            # print(rewards[traffic])
        with torch.no_grad():
            for state in states:
                action = actor(torch.Tensor(states[state]).to(device))
                action = action.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
                actions[state] = action

        this_action=0
        for traffic in actions:
            this_action = actions[traffic].copy()
            this_action[0] = 180 + 180*this_action[0]
            this_action[1] = 20500* + 20500*this_action[1]
            this_action[2] = 245 + (105*this_action[2])
            traffic.set_actions(this_action)


        final_terminated = not envs.step()
    envs.close()