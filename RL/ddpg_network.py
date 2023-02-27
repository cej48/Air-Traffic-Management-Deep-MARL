import tensorflow as tf
from typing import Tuple


class DDPGNetwork:
    def __init__(
        self,
        observation_shape: Tuple[int],
        action_shape: Tuple[int],
        scale_factor: float = 1.0,
        model: tf.keras.Model = None,
    ) -> None:
        self.observation_shape: Tuple[int] = observation_shape
        self.action_shape: Tuple[int] = action_shape
        self.scale_factor: float = scale_factor

    @classmethod
    def from_network(cls, other_network: "DDPGNetwork") -> "DDPGNetwork":
        if not isinstance(other_network, DDPGNetwork):
            raise TypeError("other_network must be an instance of DDPGNetwork.")
        new_network = cls(
            observation_shape = other_network.observation_shape,
            action_shape = other_network.action_shape,
            scale_factor = other_network.scale_factor,
            model = tf.keras.models.clone_model(other_network.model),
        )
        new_network.model.set_weights(other_network.model.get_weights())
        return new_network

    def _init_base_model(self) -> tf.keras.models.Sequential:
        pass

    def predict(self) -> Tuple[float]:
        raise NotImplementedError("Should be overridden in subclass.")
        

    @tf.function
    def merge_observation_action_batches(self, observations, actions):
        return tf.concat([observations, actions], axis=1)
        
    @tf.function
    def update_target(self, weights, beta):
        for (a, b) in zip(self.model.variables, weights):
            a.assign(b * beta + a * (1 - beta))

    @tf.function
    def assign_variables(self, variables):
        for (a,b) in zip(self.model.variables, variables):
            a.assign(b)
            
    @tf.function
    def get_nn_variables(self):
        return self.model.variables