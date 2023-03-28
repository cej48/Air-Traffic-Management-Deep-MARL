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
from replay_buffer import ReplayBuffer
import gym

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
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
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
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class EnvWrap():
    def __init__(self):
        self.env = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 1,0,0)
        self.action_space = gym.spaces.Box(low=0, high=360, shape = (3,))
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape  = np.array(self.env.traffic[0].get_observation()).shape)
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
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    #def __init__(self, capacity, batch_size, buffer_ready, states, actions):
    rb = ReplayBuffer(args.buffer_size, args.batch_size, 1000, envs.observation_space.shape, envs.action_space.shape)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    states = {i : i.get_observation() for i in envs.env.traffic}  
    actions = {i : [] for i in envs.env.traffic}
    rewards = {i : 0 for i in envs.env.traffic}
    terminated = {i : False for i in envs.env.traffic}


    # for each step
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            for state in states:
                actions[state] = np.array(envs.single_action_space.sample())
        else:
            with torch.no_grad():
                for state in states:
                    action = actor(torch.Tensor(obs).to(device))
                    action += torch.normal(0, actor.action_scale * args.exploration_noise)
                    action = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
                    actions[state] = action
        for traffic in actions:
            actions[traffic][0] = 180 + 180*actions[traffic][0]
            traffic.set_actions(actions[traffic])

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, dones, infos = envs.step(actions)ffinal_terminated = not self.env.step(inal_terminated = not self.env.step()
        final_terminated = not envs.step()

        if final_terminated:
            envs.reset()
            undiscounted_reward=0
            states = {i : i.get_observation() for i in envs.env.traffic}  
            actions = {i : [] for i in envs.env.traffic}
            rewards = {i : 0 for i in envs.env.traffic}
            terminated = {i : False for i in envs.env.traffic}
            continue


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # for info in infos:
        #     if "episode" in info.keys():
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        for traffic in terminated:
            terminated[traffic] = True if traffic not in envs.env.traffic else False
        states = {traffic : states[traffic] for traffic in envs.env.traffic}
        actions = {traffic : actions[traffic] for traffic in envs.env.traffic}
        rewards = {traffic : rewards[traffic] for traffic in envs.env.traffic}

        for traffic in envs.env.traffic:
            rewards[traffic] = traffic.reward
        
        observation = {i : i.get_observation() for i in envs.env.traffic}
        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        for traffic in states:
            if (terminated[traffic]):
                rb.insert(np.array(states[traffic]), actions[traffic], rewards[traffic], np.array(states[traffic]), terminated[traffic])
            else:
                rb.insert(np.array(states[traffic]), actions[traffic], rewards[traffic], np.array(observation[traffic]), terminated[traffic])

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        states = observation
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # data = rb.sample(args.batch_size)
            learn_state, learn_action, learn_reward, learn_observation, learn_terminated = rb.sample()
            with torch.no_grad():
                clipped_noise = (torch.randn_like(learn_action, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(learn_observation) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(learn_observation, next_state_actions)
                qf2_next_target = qf2_target(learn_observation, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = learn_reward.flatten() + (1 - learn_terminated.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(learn_state, learn_action).view(-1)
            qf2_a_values = qf2(learn_state, learn_action).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(learn_state, actor(learn_state)).mean()
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

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()