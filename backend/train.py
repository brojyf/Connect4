import torch
from Agent import Agent
from Connect4Env import Connect4Env

def train():
    env = Connect4Env()
    agent = Agent()

    num_episodes = 100000  # Training Games
    epsilon_start = 1.0
    epsilon_end = 0.005
    epsilon_decay = 0.99995  # Decay
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            # ReplayBuffer
            agent.push_experience(state, action, reward, next_state, done)

            state = next_state
            agent.update()

            # Change epsilon
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay

        # Agent update
        for _ in range(10):
            agent.update()

        # Output training process
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Epsilon={epsilon:.4f}")

    # Complete training
    torch.save(agent.policy_net.state_dict(), "connect4_dqn.pth")
    print("Training is done, save to connect4_dqn.pth。")

if __name__ == "__main__":
    # Choose mps
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("Using device:", device)
    train()
