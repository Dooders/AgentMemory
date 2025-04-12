import argparse
import os.path  # Add minimal os.path import for file operations
import random
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MODEL_PATH = "cartpole_model.pth"


# DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.fc(x)


# Experience replay
memory = deque(maxlen=MEMORY_SIZE)

# Environment setup
env = None  # Define env at global level
model = None  # Define model at global level
target_model = None  # Define target_model at global level
optimizer = None  # Define optimizer at global level
criterion = None  # Define criterion at global level


def select_action(state, epsilon, model, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    return q_values.argmax().item()


def train_step(model, target_model, optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)

    q_values = model(states).gather(1, actions)
    next_q_values = target_model(next_states).max(1, keepdim=True)[0].detach()
    target = rewards + (GAMMA * next_q_values * (1 - dones))

    loss = criterion(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Training and visualization function definitions
def train():
    # Environment setup
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    epsilon = EPSILON_START

    # Training loop
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            action = select_action(state, epsilon, model, env)
            next_state, reward, done, truncated, _ = env.step(action)
            memory.append((state, action, reward, next_state, done or truncated))
            state = next_state
            total_reward += reward

            train_step(model, target_model, optimizer, criterion)

            if done or truncated:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Update target network every 10 episodes
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    env.close()

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model, state_size, action_size


def visualize_model(num_episodes=5, model=None, state_size=None, action_size=None):
    """
    Visualize the performance of the trained model
    """
    print("\nVisualizing trained model performance...")

    if model is None:
        # Environment setup just to get dimensions
        temp_env = gym.make("CartPole-v1")
        state_size = temp_env.observation_space.shape[0]
        action_size = temp_env.action_space.n
        temp_env.close()

        # Load the model
        model = DQN(state_size, action_size)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    for episode in range(num_episodes):
        # Render mode needs to be set to 'human' for visualization
        test_env = gym.make("CartPole-v1", render_mode="human")
        state, _ = test_env.reset()
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Visualization Episode {episode+1}, Total Reward: {total_reward}")
        test_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CartPole DQN")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to visualize"
    )

    args = parser.parse_args()

    # Default behavior: if no args are provided, do both train and visualize
    if not args.train and not args.visualize:
        args.train = True
        args.visualize = True

    if args.train:
        model, state_size, action_size = train()
        if args.visualize:
            visualize_model(args.episodes, model, state_size, action_size)
    elif args.visualize:
        visualize_model(args.episodes)
