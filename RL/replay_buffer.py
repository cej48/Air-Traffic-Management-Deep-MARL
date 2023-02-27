
import numpy as np
import tensorflow as tf
from random import randint

class ReplayBuffer:
    def __init__(self, capacity, batch_size, buffer_ready, states, actions):
        self.capacity=capacity
        self.batch_size = batch_size

        self.state_buffer = np.empty((self.capacity, states[0]))
        self.action_buffer = np.empty((self.capacity, actions[0]))
        self.reward_buffer = np.empty((self.capacity, 1))
        self.observation_buffer = np.empty((self.capacity,states[0]))
        self.terminated_buffer = np.empty((self.capacity, 1))

        self.state_buffer.fill(0) 
        self.action_buffer.fill(0) 
        self.reward_buffer.fill(0)
        self.observation_buffer.fill(0)
        self.terminated_buffer.fill(0) # Prefer static mem usage
        
        self.current_capacity=0
        self.index = 0
        self.buffer_ready = buffer_ready

    def insert(self, state, action, reward, observation, terminated):

        self.state_buffer[self.index]=state
        self.action_buffer[self.index]=action
        self.reward_buffer[self.index]=reward
        self.observation_buffer[self.index]=observation
        self.terminated_buffer[self.index]=terminated
    
        self.index+=1
        if self.current_capacity<self.capacity:
            self.current_capacity+=1
        if self.index>=self.capacity:
            self.index=0

        
    def is_sample_ready(self):
        return True if self.current_capacity>self.buffer_ready else False
    
    # @tf.function
    def sample(self):
        sample_size = self.batch_size if self.current_capacity > self.batch_size else self.current_capacity
        sample_point = randint(0, self.current_capacity - sample_size)
        # sample_point = tf.experimental.numpy.random.randint(minval)
        sample_end = sample_point + sample_size

    
        sample =   (tf.convert_to_tensor(self.state_buffer[sample_point:sample_end], dtype=tf.float32), 
                    tf.convert_to_tensor(self.action_buffer[sample_point:sample_end], dtype=tf.float32),
                    tf.convert_to_tensor(self.reward_buffer[sample_point:sample_end], dtype=tf.float32),
                    tf.convert_to_tensor(self.observation_buffer[sample_point:sample_end], dtype=tf.float32), 
                    tf.convert_to_tensor(self.terminated_buffer[sample_point:sample_end], dtype=tf.float32))
        return sample