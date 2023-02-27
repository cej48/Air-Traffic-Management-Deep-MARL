import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple
from ddpg_network import DDPGNetwork

class CriticNetwork(DDPGNetwork):
    def __init__(
        self,
        observation_shape: Tuple[int],
        action_shape: Tuple[int],
        scale_factor: float = 1.0,
        learning_rate : float = 0.001,
        model: tf.keras.Model = None,
    ) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.input_shape = (action_shape[0] + observation_shape[0],)
        self.output_shape = (1,)
        super(CriticNetwork, self).__init__(observation_shape, action_shape, scale_factor)
        self.model: tf.keras.Model = model if model is not None else self.__init_model()

    def __init_model(self) -> tf.keras.Model:

        model: tf.keras.models.Sequential = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape, name="input"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        # model.add(tf.keras.layers.LayerNormalization(axis=1))

        model.add(tf.keras.layers.Dense(units=self.output_shape[0], activation="linear", name="output"))
        model.compile(run_eagerly=True,  optimizer=self.optimizer)
        return model

    @tf.function
    def predict_batch_raw(self, batch, training):
        return self.model(batch, training=training)
    

    @tf.function 
    def gradient_descent(self, target_critic, target_observation_action, learn_state_actions, learn_terminated, learn_rewards, gamma):

        learn_rewards = tf.reshape(tf.cast(learn_rewards, float), (len(learn_rewards),1)) # Reshaped for correct matrix *
        learn_terminated = tf.reshape(1 - tf.cast(learn_terminated, dtype=tf.float32), (len(learn_terminated), 1)) # Mask, s.t if terminated, value is 0
        y = learn_rewards + gamma * (target_critic.predict_batch_raw(target_observation_action, training=False))*learn_terminated

        with tf.GradientTape() as tape:
            critic_val = self.model(learn_state_actions, training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y-critic_val))
            # critic_loss = tf.math.square(y-critic_val)
        
        critic_gradient = tape.gradient(critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_gradient, self.model.trainable_variables))
        


            
