# Christopher Jones 2023

import random
import time
from distutils.util import strtobool

#  import gym
import sys
sys.path.append("build/")
import PyATMSim # import our ATMSim library

from random import randint
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

load_from_file = True

batch_size = 1024
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

# Define critic and actor networks.
class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
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
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape))
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


# Wrap our ATMSim to work with gym boxes.
class EnvWrap():
    def __init__(self):
        self.env = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0,25)
        self.env.step()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape = (3,))
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape  = np.array(self.env.traffic[0].get_observation()).shape)
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
    envs = EnvWrap()

    if load_from_file: # Load networks from a saved file
        learning_starts=0

        actor = Actor(envs)
        actor.load_state_dict(torch.load("./models/actor.pt"))
        actor.to(device)

        target_actor = Actor(envs)
        target_actor.load_state_dict(torch.load("./models/actor_targ.pt"))
        target_actor.to(device)

        qf1 = Critic(envs)
        qf1.load_state_dict(torch.load("./models/qf1.pt"))
        qf1.to(device)

        qf2 = Critic(envs)
        qf2.load_state_dict(torch.load("./models/qf2.pt"))
        qf2.to(device)

        qf2_target = Critic(envs)
        qf2_target.load_state_dict(torch.load("./models/qf2_targ.pt"))
        qf2_target.to(device)

        qf1_target = Critic(envs)
        qf1_target.load_state_dict(torch.load("./models/qf1_targ.pt"))
        qf1_target.to(device)
    
    else:   # Reset output csv
        with open("output.csv", "w") as file:
            file.write("step, arrivals_sum, infringement_sum, reward \n")
        actor = Actor(envs).to(device)
        qf1 = Critic(envs).to(device)
        qf2 = Critic(envs).to(device)
        qf1_target = Critic(envs).to(device)
        qf2_target = Critic(envs).to(device)
        target_actor = Actor(envs).to(device)

        target_actor.load_state_dict(actor.state_dict())
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=learning_rate)
    qf1_optimizer = optim.Adam(list(qf1.parameters()), lr = learning_rate)
    qf2_optimizer = optim.Adam(list(qf2.parameters()), lr = learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=learning_rate)

    envs.observation_space.dtype = np.float32
    # Set up buffer on GPU 
    buffer = ReplayBuffer(
        buffer_size,
        envs.observation_space,
        envs.action_space,
        device,
        handle_timeout_termination=False,
        optimize_memory_usage=False,
        n_envs=1)
    start_time = time.time()
    obs = envs.reset()

    # Store all agent info in dict for easy access.
    states = {i : i.get_observation() for i in envs.env.traffic}  
    actions = {i : [] for i in envs.env.traffic}
    rewards = {i : 0 for i in envs.env.traffic}
    terminated = {i : False for i in envs.env.traffic}
    reward = 0
    noise = torch.normal(0, actor.action_scale * exploration_noise)


    for global_step in range(total_timesteps):
        start_time = time.time()
        new_state = {}
        for traffic in envs.env.traffic:
            # If traffic not in the states, add it.
            if not traffic.terminated:
                if traffic in states:
                    new_state[traffic] = states[traffic] 
                else:
                    new_state[traffic] = traffic.get_observation()

        states = new_state
        actions = {}
        rewards = {}
        terminated = {}
        observation = {}
        if global_step < learning_starts:
            for state in states:
                actions[state] = np.array(envs.action_space.sample())
        else:
            # Get all actions.
            with torch.no_grad():
                for state in states:
                    action = actor(torch.Tensor(states[state]).to(device))
                    action += torch.normal(0, actor.action_scale * exploration_noise)
                    action = action.cpu().numpy().clip(envs.action_space.low, envs.action_space.high)
                    actions[state] = action


        this_action=0
        # Set the actions.
        for traffic in actions:
            this_action = actions[traffic].copy()
            this_action[0] = 180 + 180*this_action[0]
            this_action[1] = 20500* + 20500*this_action[1]
            this_action[2] = 245 + (105*this_action[2])
            traffic.set_actions(this_action)

        #Step env
        final_terminated = not envs.step()

        for state in list(states): # remove traffic that has timed out. (non terminal state, but may be stuck)
            if state not in envs.env.traffic:
                states.pop(state)

        for traffic in states: # Set terminated values.
            terminated[traffic] = traffic.terminated
        

        for traffic in states:
            reward+=traffic.reward
            rewards[traffic] = -170 + ((traffic.reward))
            if rewards[traffic]>0:
                print("reward scheme failure")

        # Get observations from envs. 
        observation = {i : i.get_observation() for i in envs.env.traffic}
        
        # Add experience.
        for traffic in states:
            buffer.add(states[traffic], observation[traffic], actions[traffic], rewards[traffic], terminated[traffic], [])
        
        # Learning.
        states = observation
        if global_step > learning_starts:
            #Get sample
            data = buffer.sample(batch_size)
            with torch.no_grad():
                #Clip
                clipped_noise = (torch.randn_like(data.actions, device=device) * policy_noise).clamp(
                    -noise_clip, noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.action_space.low[0], envs.action_space.high[0]
                )

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * gamma * (min_qf_next_target).view(-1)


            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)

            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % policy_frequency == 0 and global_step > 5000:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # save models.
        if global_step % 100 == 0:
            if global_step % 500==0:
                with open("output.csv", "a+") as file:
                    file.write(f"{envs.env.total_steps},{envs.env.total_arrivals}, {envs.env.total_infringements}, {reward}\n")
                    reward=0
            if global_step % 10000 ==0:
                torch.save(qf1.state_dict(), "./models/qf1.pt")
                torch.save(qf2.state_dict(), "./models/qf2.pt")
                torch.save(qf1_target.state_dict(), "./models/qf1_targ.pt")
                torch.save(qf2_target.state_dict(), "./models/qf2_targ.pt")
                torch.save(actor.state_dict(), "./models/actor.pt")
                torch.save(target_actor.state_dict(), "./models/actor_targ.pt")

            end_time = time.time()
            print(f"Time: {end_time-start_time}")
