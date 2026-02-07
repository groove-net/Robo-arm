import os
import random
import numpy as np
from inhand_env import CanRotateEnv, MAX_EPISODE_STEPS, MacroAction
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Initialize the Environment
env = CanRotateEnv(render_mode="human")

# DQN Architecture
class DQNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, size=100_000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(ns),
            np.array(d, dtype=np.float32)
        )

# Create Seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Epsilon-greedy action selection
# -------------------------
def select_action(model, obs, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = model(obs_t)
    return int(torch.argmax(q, dim=1).item())

# -------------------------
# Training step
# -------------------------
def train_step(model, target_model, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch

    states      = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions     = torch.tensor(actions, dtype=torch.int64)
    rewards     = torch.tensor(rewards, dtype=torch.float32)
    dones       = torch.tensor(dones, dtype=torch.float32)

    # Q(s, a)
    q_values = model(states)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target: r + γ max_a' Q_target(s',a')
    with torch.no_grad():
        next_q = target_model(next_states)
        max_next_q = next_q.max(dim=1)[0]
        target = rewards + (1 - dones) * gamma * max_next_q

    # Loss
    loss = nn.MSELoss()(q_sa, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# -------------------------
# Main Deep Q-Netwrok training loop
# -------------------------
EPISODES = 5000
def train_dqn():
    # Delete old log file if it exists
    log_path = "training_log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)

    obs, _ = env.reset()
    obs_dim = len(obs)
    num_actions = env.action_space.n

    model = DQNetwork(obs_dim, num_actions)
    target_model = DQNetwork(obs_dim, num_actions)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    buffer = ReplayBuffer(size=200_000)

    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.0005
    gamma = 0.99

    macro_steps = 10
    target_update_interval = 3000  # env steps
    step_count = 0

    # Open log file
    log_file = open(log_path, "w")

    for episode in range(EPISODES):

        obs, _ = env.reset()
        episode_reward = 0

        for t in range(MAX_EPISODE_STEPS):

            # --- Select action ---
            action = select_action(model, obs, epsilon, num_actions)

            # --- Execute macro-action ---
            total_reward = 0.0
            done = False
            for _ in range(macro_steps): 
                next_obs, reward, terminated, truncated, _ = env.step(MacroAction(action)) 
                total_reward += reward 
                step_count += 1
                if terminated or truncated:
                    done = True
                    break

            buffer.add(obs, action, total_reward, next_obs, done)

            obs = next_obs
            episode_reward += total_reward

            # --- Train ---
            if len(buffer.buffer) > batch_size:
                batch = buffer.sample(batch_size)
                loss = train_step(model, target_model, optimizer, batch, gamma)

            # --- Update target network ---
            if step_count % target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

            if done:
                break

        # Decay epsilon once per episode
        epsilon = max(epsilon_min, epsilon * (1 - epsilon_decay))

        log_line = f"Episode {episode} | Reward {episode_reward:.3f} | eps={epsilon:.3f}\n"
        print(log_line, end="")
        log_file.write(log_line)

        if (episode + 1) % 1000 == 0:
            torch.save(model.state_dict(), f"dqn_ep{episode + 1}.pt")

    log_file.close()
    env.close()
    
if __name__ == "__main__":
    train_dqn()