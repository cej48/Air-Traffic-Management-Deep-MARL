import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Tuple
from ddpg_network import DDPGNetwork
from actor import ActorNetwork
from critic import CriticNetwork
import PyATMSim
import copy
# from gym.envs.mujoco.humanoid_v4 import HumanoidEnv


# class ExtendedHumanoidV4(HumanoidEnv):
#     def reset(self):
#         pass

class MetaPolicy(DDPGNetwork):
    def __init__(
        self,
        observation_shape : Tuple[int],
        action_shape: Tuple[int],
        learning_rate : float = 0.001,
        model: tf.keras.Model = None,
        peak_ahead : int = 3,
        master_sim = None
    ) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        self.input_shape = (observation_shape[0],)
        self.output_shape = action_shape
        super(MetaPolicy, self).__init__(action_shape, action_shape,1)

        self.model: tf.keras.Model = model if model is not None else self.__init_model()

        self.peak_ahead=peak_ahead

        self.d_0_observations = np.zeros((self.peak_ahead, self.input_shape[0]))
        self.d_0_actions = np.zeros((self.peak_ahead, self.action_shape[0]))

        self.d_o_a_observations = np.zeros((peak_ahead, self.input_shape[0]))
        self.d_1_actions = np.zeros((peak_ahead, self.action_shape[0]))
        self.d_1_observations = np.zeros((peak_ahead, self.input_shape[0]))
        print("here1")
        self.mp_env = PyATMSim.ATMSim("environment_boundaries.json","airport_data.json", 0,0,0)
        print("here2")
        # print(dataset)

    def __init_model(self) -> tf.keras.Model:
        model: tf.keras.models.Sequential = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape, name="input"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        # model.add(tf.keras.layers.LayerNormalization(axis=1))

        model.add(tf.keras.layers.Dense(units=self.action_shape[0]*2, activation="tanh", name="output"))

        model.compile(run_eagerly=True,  optimizer=self.optimizer)
        return model

    @tf.function
    def update_model(self, d_0_observations, d_0_actions, d_o_a_val, d_1_val):
        with tf.GradientTape() as tape:

            means, variances = tf.split(self.model(d_0_observations), num_or_size_splits = 2, axis = 1)

            distribution = tfp.distributions.Normal(loc = means, scale = tf.math.abs(variances))
            probabilities = distribution.log_prob(d_0_actions)

            value = tf.cast(d_1_val-d_o_a_val,tf.float32)
            loss = -tf.math.reduce_sum(probabilities)*value
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def gen_action(self, state):
        out = self.model(state, True) 
        distribution = tfp.distributions.Normal(loc = out[0][0], scale = out[0][1])
        return tf.clip_by_value(distribution.sample(self.action_shape), -1,1)


    def generate_experience(self, actor, critic, observation, 
                            actor_policy_reward, master_sim, buffer):
        self.mp_env.copy_from_other(master_sim)
        return

        d_0 = observation
        state = observation
        for i in tf.range(0,self.peak_ahead):
            action = self.gen_action(tf.expand_dims(tf.convert_to_tensor(d_0), 0))
            self.d_0_actions[i] = action
            self.d_0_observations[i] = np.array(d_0)
            d_0, reward, terminated,*_ = self.mp_env.step(action)
            buffer.insert(np.array(state), action, reward, np.array(d_0), terminated) 
            state = d_0
            if terminated:state, *_ = self.mp_env.reset()
        d_0_tensor = tf.convert_to_tensor(self.d_0_observations)
        value_restore = actor.get_nn_variables() # we do not want to update our actual actor based on this
        actor.gradient_ascent(critic, tf.cast(d_0_tensor, tf.float32))


        self.set_mp_env_state(observation_qpos, observation_qvel)

        d_1 = observation
        state = observation
        d_1_reward = 0
        for i in range(self.peak_ahead):
            action = actor.model(tf.expand_dims(tf.convert_to_tensor(d_1), 0), False)[0]
            d_1, reward, terminated,*_ = self.mp_env.step(action)
            buffer.insert(np.array(state), action, reward, np.array(d_1), terminated)
            state = d_1
            d_1_reward+=reward
            if terminated:state, *_ = self.mp_env.reset()

        self.update_model(tf.cast(d_0_tensor, tf.float32), tf.cast(self.d_0_actions, tf.float32), actor_policy_reward, d_1_reward)
        actor.assign_variables(value_restore)
