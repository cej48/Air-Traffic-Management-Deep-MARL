import tensorflow as tf
import numpy as np
from typing import Tuple
from ddpg_network import DDPGNetwork
from critic import CriticNetwork


class ActorNetwork(DDPGNetwork):
    def __init__(
        self,
        observation_shape: Tuple[int],
        action_shape: Tuple[int],
        scale_factor: float = 1.0,
        learning_rate : float = 0.001,
        model: tf.keras.Model = None,
    ) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.input_shape = observation_shape
        self.output_shape = action_shape
        super(ActorNetwork, self).__init__(observation_shape, action_shape, scale_factor)
        self.model: tf.keras.Model = model if model is not None else self.__init_model()

    def __init_model(self) -> tf.keras.Model:
        model: tf.keras.models.Sequential = tf.keras.models.Sequential()
        print(self.input_shape)
        model.add(tf.keras.layers.Input(shape=self.input_shape, name="input"))
        model.add(tf.keras.layers.Dense(units=128, activation="sigmoid"))
        model.add(tf.keras.layers.Dense(units=128, activation="sigmoid"))
        # model.add(tf.keras.layers.LayerNormalization(axis=1))

        model.add(tf.keras.layers.Dense(units=self.action_shape[0], activation="tanh", name="output"))

        model.compile(run_eagerly=True,  optimizer=self.optimizer)
        return model

    @tf.function
    def gradient_ascent(self, critic, observation):
        with tf.GradientTape() as tape:
            
            this_action = self.model(observation, training=True)

            actor_loss = critic.predict_batch_raw(critic.merge_observation_action_batches(observation, this_action), True)
            actor_loss = -tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_gradient, self.model.trainable_variables))


