import gymnasium as gym
import tensorflow as tf
from pathlib import Path
from ddpg_agent import DDPGAgent



def main() -> None:
    env = gym.make("Ant-v4", render_mode="human")#, ctrl_cost_weight=0.4)
    timestamp = "2023-01-05 12-54-12"
    path: Path = Path(f"./models/{timestamp}")
    agent = DDPGAgent(env=env, beta=0.005, gamma=0.99, sample_size=128)#, path=path) # Higher is less stable for beta.
    for i in range(1):
        agent.train(episodes=120000, max_steps=250)
        agent.save()
    # for _ in range(20):
    #     agent.run(max_steps=500)

def run() -> None:
    env = gym.make("InvertedDoublePendulum-v4", render_mode="human")

    timestamp = "2022-12-29 07-44-32"
    path: Path = Path(f"./models/{timestamp}")

    agent = DDPGAgent(env=env, beta=0.5, gamma=0.01, sample_size=10, path=path)

    observation, info = env.reset()
    while True:
        observation=tf.convert_to_tensor(tf.expand_dims(observation,0))
        action = list(agent.actor.predict(observation, False))
        observation, reward, terminated, truncated, info = env.step(action[0])
        # if terminated:
        #     env.reset()

    env.close()

if __name__ == "__main__":
    main()
    # run()
