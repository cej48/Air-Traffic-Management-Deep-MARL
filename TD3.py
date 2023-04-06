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
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.0005,
        help="target smoothing coefficient (default: 0.005)")
    # parser.add_argument("--batch-size", type=int, default=512,
    #     help="the batch size of sample from the reply memory")
    parser.add_argument("--batch-size", type=int, default=8196,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.2,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=100e3,
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
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
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
        x = torch.tanh(self.fc_mu(x))
        
        return x * self.action_scale + self.action_bias


class EnvWrap():
    def __init__(self):
        self.env = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0)
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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = EnvWrap()
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    qf1_optimizer = optim.Adam(list(qf1.parameters()), lr = args.learning_rate)
    qf2_optimizer = optim.Adam(list(qf2.parameters()), lr = args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    #def __init__(self, capacity, batch_size, buffer_ready, states, actions):
    # rb = ReplayBuffer(args.buffer_size, args.batch_size, 1000, envs.observation_space.shape, envs.action_space.shape)
    buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        optimize_memory_usage=False,
        n_envs=1)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    states = {i : i.get_observation() for i in envs.env.traffic}  
    actions = {i : [] for i in envs.env.traffic}
    rewards = {i : 0 for i in envs.env.traffic}
    terminated = {i : False for i in envs.env.traffic}

    noise = torch.normal(0, actor.action_scale * args.exploration_noise)
    # for each step
    for global_step in range(args.total_timesteps):
        start_time = time.time()
        # ALGO LOGIC: put action logic here
        new_state = {}

        for traffic in envs.env.traffic:
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

        if global_step < args.learning_starts:
            for state in states:
                actions[state] = np.array(envs.single_action_space.sample())
        else:
            with torch.no_grad():
                for state in states:
                    action = actor(torch.Tensor(states[state]).to(device))
                    action += torch.normal(0, actor.action_scale * args.exploration_noise)
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

        for state in list(states): # remove traffic that has timed out. (non terminal state, but may be stuck)
            if state not in envs.env.traffic:
                states.pop(state)

        for traffic in states:
            terminated[traffic] = traffic.terminated

        for traffic in states:

            rewards[traffic] =(traffic.reward)
            # print(rewards  [traffic])
        observation = {i : i.get_observation() for i in envs.env.traffic}
        
        for traffic in states:
            buffer.add(states[traffic], observation[traffic], actions[traffic], rewards[traffic], terminated[traffic], [])
        
        if final_terminated:
            envs.reset()
            undiscounted_reward=0
            states = {i : i.get_observation() for i in envs.env.traffic}  
            actions = {i : [] for i in envs.env.traffic}
            rewards = {i : 0 for i in envs.env.traffic}
            terminated = {i : False for i in envs.env.traffic}
            continue

        states = observation
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # for i in range(len(envs.env.traffic)):
                data = buffer.sample(args.batch_size)
                with torch.no_grad():
                    clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * target_actor.action_scale

                    next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                        envs.single_action_space.low[0], envs.single_action_space.high[0]
                    )

                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    # print(qf1_next_target)

                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                    # print(next_q_value)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)


                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                    # print(actor_loss)
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    # print(qf2_target.state_dict()[list(qf2_target.state_dict())[1]])
                    # print(target_actor.state_dict()[list(target_actor.state_dict())[2]])

        if global_step % 100 == 0:

            end_time = time.time()
            print(f"Time: {end_time-start_time}")



    envs.close()
    writer.close()