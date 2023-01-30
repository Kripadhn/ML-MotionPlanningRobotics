import gym
import numpy as np

# Load the environment
env = gym.make("Pendulum-v1")

# Define the start and goal states
start = env.reset()
goal = np.array([np.pi, 0])

# Define the RRT function
def rrt(start, goal, env, max_steps=1000, step_size=0.1):
    nodes = [start]
    for i in range(max_steps):
        random_state = env.observation_space.sample()
        closest_node = nodes[np.argmin([np.linalg.norm(node - random_state) for node in nodes])]
        new_node = closest_node + (random_state - closest_node) / np.linalg.norm(random_state - closest_node) * step_size
        new_node = np.clip(new_node, env.observation_space.low, env.observation_space.high)
        nodes.append(new_node)
        if np.linalg.norm(new_node - goal) < step_size:
            break
    return nodes

# Find the path using RRT
path = rrt(start, goal, env)

# Visualize the path
env.render()
for state in path:
    env.state = state
    env.render()
