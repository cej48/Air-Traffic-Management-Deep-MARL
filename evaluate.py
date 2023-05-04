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

batch_size = 512
tau = 0.005
policy_frequency = 2
noise_clip = 0.5
learning_starts = 5e3
exploration_noise = 0.2
policy_noise = 0.2
gamma=0.999
buffer_size = int(1e6)
learning_rate = 1e-4
total_timesteps = int(1e8)
torch_deterministic = True
seed = 1

cuda = True

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = EnvWrap()
    actor = Actor(envs)
    # device = "cpu"
    if device == torch.device("cuda"):
        actor.load_state_dict(torch.load("./models/actor.pt"))
        actor.to(device)
    else:
        actor.load_state_dict(torch.load("./models/actor.pt", map_location=device))

    envs.single_observation_space.dtype = np.float32
    obs = envs.reset()

    states = {i : i.get_observation() for i in envs.env.traffic}  
    actions = {i : [] for i in envs.env.traffic}

    aircraft = envs.env.traffic[0]
    prev = states[aircraft][2]*41000

    index = 0
    max_aircraft = 50

    altitudes = {i : [] for i in range(max_aircraft)}
    speeds = {i : [] for i in range(max_aircraft)}
    distances = {i : [] for i in range(max_aircraft)}


    lifespan_file = open("./results/lifespans.csv", "w+")

    while index < max_aircraft:
        prev = list(states)
        states = {i : i.get_observation() for i in envs.env.traffic}
        actions={}

        for traffic in states:
            reward=-170 + traffic.reward
            if reward>0:
                print(reward)
            if traffic.terminated:
                lifespan_file.write(str(traffic.lifespan)+'\n')
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
        if aircraft not in list(states):
            for i in list(states):
                if i not in prev:
                    aircraft = i
                    index+=1
                    # altitude_file.write("NEW \n")
                    # speed_file.write("NEW\n")
                    # distance_file.write("NEW\n")
        else:
            altitudes[index].append(str(states[aircraft][2]*41000))
            speeds[index].append(str(states[aircraft][4]*350))
            distances[index].append(str(aircraft.distance_to))

        # prev = states[aircraft][2]*41000
        # altitude_file.write(str(states[aircraft][2]*41000) + '\n')
        # speed_file.write(str(states[aircraft][4]*350) + '\n')
        # distance_file.write(str(aircraft.distance_to)+ '\n')


    altitude_file = open("./results/altitudes.csv", "w+")
    speed_file = open("./results/speeds.csv", "w+")
    distance_file = open("./results/distance.csv", "w+")
    infringements = open(open("./results/infringements.txt", "w+"))

    max_length = lambda dict : max([len(i) for i in dict.values()]) 
    for i in range(max_length(altitudes)):
        string = ""
        for ac in range(max_aircraft):
            if i < len(altitudes[ac]):
                string += altitudes[ac][i]
            string+=","
        altitude_file.write(string + '\n')

    for i in range(max_length(speeds)):
        string = ""
        for ac in range(max_aircraft):
            if i < len(speeds[ac]):
                string += speeds[ac][i]
            string+=","
        speed_file.write(string + '\n')

    for i in range(max_length(distances)):
        string = ""
        for ac in range(max_aircraft):
            if i < len(distances[ac]):
                string += distances[ac][i]
            string+=","
        distance_file.write(string + '\n')
    infringements.write("total: " + str(envs.env.total_infringements))
    infringements.write("near: " + str(envs.env.total_near_infringements))
    