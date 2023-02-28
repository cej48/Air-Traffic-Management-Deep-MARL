import random
import tensorflow as tf
import numpy as np
import gc
import copy

import time

from noise import OrnsteinUhlenbeck

from replay_buffer  import ReplayBuffer
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from actor import ActorNetwork
from critic import CriticNetwork
from meta_policy import MetaPolicy

import PyATMSim

class DDPGAgent:
    def __init__(
        self,
        env,
        beta: float,
        gamma: float,
        sample_size : int,
        path: Path = None,
    ) -> None:

        self.observation_space = np.array(env.traffic[0].get_observation())
        self.action_space = np.zeros(3)

        self.env = env

        if path is not None:
            self.actor: ActorNetwork = ActorNetwork(
                observation_shape = self.observation_space.shape,
                action_shape = self.action_space.shape,
                scale_factor = self.action_space.high[0],
                learning_rate=0.0001,
                model = tf.keras.models.load_model(path / f"actor.h5"),
            )
            self.critic: CriticNetwork = CriticNetwork(
                observation_shape = self.observation_space.shape,
                action_shape = self.action_space.shape,
                learning_rate=0.001,
                model=tf.keras.models.load_model(path / f"critic.h5"),
            )
            self.target_actor: ActorNetwork = ActorNetwork(
                observation_shape = self.observation_space.shape,
                action_shape = self.action_space.shape,
                scale_factor = self.action_space.high[0],
                model = tf.keras.models.load_model(path / f"target_actor.h5"),
            )
            self.target_critic: CriticNetwork = CriticNetwork(
                action_shape = self.action_space.shape,
                observation_shape = self.observation_space.shape,
                model=tf.keras.models.load_model(path / f"target_critic.h5"),
            )
            self.meta_policy : MetaPolicy = MetaPolicy(self.action_space.shape,
                self.observation_space.shape,
                0.0001,
                model=tf.keras.models.load_model(path / f"meta_policy.h5"))
        else:

            self.actor: ActorNetwork = ActorNetwork(self.observation_space.shape, self.action_space.shape)
            self.critic: CriticNetwork = CriticNetwork(self.observation_space.shape, self.action_space.shape)
            self.target_actor: ActorNetwork = ActorNetwork.from_network(self.actor)

            self.target_critic: CriticNetwork = CriticNetwork.from_network(self.critic)

            self.meta_policy: MetaPolicy = MetaPolicy(self.observation_space.shape, self.action_space.shape, 0.001, self.env)
            # self.meta_policy.assign_variables(self.actor.model.get_weights())

        self.sample_size = sample_size

        self.buffer_ready = sample_size*2

        self.buffer = ReplayBuffer(100000, self.sample_size, self.buffer_ready, self.observation_space.shape, self.action_space.shape)
        self.beta = beta
        self.gamma = gamma
        
    @tf.function
    def ddpg_updates(self, learn_state, learn_action, learn_reward, learn_observation, learn_terminated):
        # print("Tracing DDPG updates")
        target_actor_action = self.target_actor.model(learn_observation, True)

        target_observation_action = self.critic.merge_observation_action_batches(observations=learn_observation, actions=target_actor_action)

        learn_state_actions = self.critic.merge_observation_action_batches(observations=learn_state, actions=learn_action)
        
        self.critic.gradient_descent(target_critic=self.target_critic, target_observation_action=target_observation_action, 
                                    learn_state_actions=learn_state_actions, learn_terminated=learn_terminated, learn_rewards=learn_reward, 
                                    gamma=self.gamma) #included gamma to support variable gamma training.

        self.actor.gradient_ascent(critic=self.critic, observation=learn_state)

        self.target_critic.update_target(self.critic.get_nn_variables(),self.beta)
        self.target_actor.update_target(self.actor.get_nn_variables(),self.beta)

    def train(self, episodes: int, max_steps: int) -> None:
        undiscounted_rewards = [0]

        sd = 1
        clip = 0.7

        undiscounted_reward = 0
        
        states = {i : i.get_observation() for i in self.env.traffic}  
        actions = {i : [] for i in self.env.traffic}
        rewards = {i : 0 for i in self.env.traffic}
        terminated = {i : False for i in self.env.traffic}

        episode = 0
        step = 0

        this_meta_run_accumulated_reward = 0
        meta_policy_observations = states

        this_step =0
        noise = tf.random.normal(self.action_space.shape, stddev=0.1, mean=0)
        while episode<episodes:
            this_step+=1
            tic = time.perf_counter()
            if (step%10):
                noise = tf.random.normal(self.action_space.shape, stddev=0.1, mean=0)
            for state in states:
                action = self.actor.model(tf.expand_dims(tf.convert_to_tensor(states[state]), 0), True)[0].numpy()
                action = np.clip(action+noise, -1,1)
                actions[state] =action
                # print(action)

            if (tf.reduce_any(tf.math.is_nan(actions[self.env.traffic[0]]))):
                print("NaN detected")

            for traffic in actions:
                actions[traffic][0] = 180 + 180*actions[traffic][0]
                traffic.set_actions(actions[traffic])

            final_terminated = not self.env.step() # sim returns 1 if running...
            
            for traffic in terminated:
                terminated[traffic] = True if traffic not in self.env.traffic else False
            states = {traffic : states[traffic] for traffic in self.env.traffic}
            actions = {traffic : actions[traffic] for traffic in self.env.traffic}
            rewards = {traffic : rewards[traffic] for traffic in self.env.traffic}

            for traffic in self.env.traffic:
                rewards[traffic] = traffic.reward
            
            observation = {i : i.get_observation() for i in self.env.traffic}

            # total_reward+= sum(reward)
            undiscounted_reward+=sum(rewards.values())
            
            for traffic in states:
                if (terminated[traffic]):
                    self.buffer.insert(np.array(states[traffic]), actions[traffic], rewards[traffic], np.array(states[traffic]), terminated[traffic])
                else:
                    self.buffer.insert(np.array(states[traffic]), actions[traffic], rewards[traffic], np.array(observation[traffic]), terminated[traffic])

            states = observation

            this_meta_run_accumulated_reward +=sum(rewards.values())
            
            # if step%3==0 and step!=0:
            #     for traffic in self.env.traffic:
            #         self.meta_policy.generate_experience(self.actor, self.critic, meta_policy_observations[traffic], this_meta_run_accumulated_reward, self.env, self.buffer)
            #         this_meta_run_accumulated_reward=0
            #         # Copy over meta policy state heree.
            #         meta_policy_observations[traffic] = tf.convert_to_tensor(observation[traffic], dtype = tf.float32)

            if self.buffer.is_sample_ready(): #dont sample the buffer if it's too small.
                learn_state, learn_action, learn_reward, learn_observation, learn_terminated = self.buffer.sample() # function below is tf graph
                self.ddpg_updates(learn_state, learn_action, learn_reward, learn_observation, learn_terminated)                
            step+=1

            toc = time.perf_counter()
            if final_terminated or this_step>max_steps:
                this_step=0

                self.env.reset()
                episode+=1
                print(f"{episode=}")
                print(f"{step=}")
                print(f"{undiscounted_reward=}")
                print(f"Step/s: {1/(toc-tic)}")
                
                undiscounted_reward=0

                states = {i : i.get_observation() for i in self.env.traffic}  
                actions = {i : [] for i in self.env.traffic}
                rewards = {i : 0 for i in self.env.traffic}
                terminated = {i : False for i in self.env.traffic}
                meta_policy_observations = states
                # state, *_ = self.env.reset()

    def write_results(self,step, cumulative_reward):
        with open("results.txt", 'w+') as f:
            for index, i in enumerate(cumulative_reward):
                f.write(f"{index}, {step[index]}, {i} \n")

    def run(self, max_steps: int) -> None:
        state, *_ = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = list(self.actor.predict(state, False))
            action = tuple(action)
            observation, reward, terminated, truncated, info = self.env.step(action)
            state = observation
            total_reward += reward
        print(f"{total_reward=}")

    def save(self, name) -> None:
        now: datetime = datetime.now()
        timestamp: str = now.strftime("%Y-%m-%d %H-%M-%S")
        path: Path = Path(f"./models/{timestamp}_{name}")
        path.mkdir(parents=True, exist_ok=True)
        self.actor.model.save(f"{path}/actor.h5")
        self.critic.model.save(f"{path}/critic.h5")
        self.target_actor.model.save(f"{path}/target_actor.h5")
        self.target_critic.model.save(f"{path}/target_critic.h5")
        self.meta_policy.model.save(f"{path}/meta_policy.h5")
        gc.collect()

