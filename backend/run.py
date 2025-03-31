import torch
import numpy as np

from Agent import Agent
from Connect4Env import Connect4Env

def play():
    # Instantiation
    env = Connect4Env()
    agent = Agent()
    # Load trained weights
    agent.policy_net.load_state_dict(torch.load("connect4_dqn_1m.pth", map_location="cpu"))
    agent.policy_net.eval()

    # Asking who's going first

    choice = input("Are you going first? (y/n): ").strip().lower()
    if choice == "y":
        human_player = 1.0
        ai_player = -1.0
    else:
        human_player = -1.0
        ai_player = 1.0

    state = env.reset()
    done = False

    while not done:
        env.render()

        if env.current_player == human_player:
            # My action
            valid_actions = [c for c in range(env.cols) if env.is_valid_action(c)]
            print(f"It's your turn，valid slots: {valid_actions}")
            col_str = input("Enter: ")
            try:
                action = int(col_str)
            except:
                action = -1  # Invalid
        else:
            # AI action
            action = agent.select_action(state, epsilon=0.0)
            print(f"AI choose {action}")

        next_state, reward, done, info = env.step(action)
        state = next_state

        if info.get("invalid", False):
            # Invalid action
            print("Invalid action！Game over。")
            break

        if done:
            env.render()
            # Check who wins
            _, winner = env.check_done()
            if winner == human_player:
                print("Congrats! You win!")
            elif winner == ai_player:
                print("AI wins!")
            else:
                print("Tie!")
            break

if __name__ == "__main__":
    play()
