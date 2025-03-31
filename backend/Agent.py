import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from DQN import DQN
from ReplayBuffer import ReplayBuffer

class Agent:
    def __init__(self, gamma=0.99, lr=1e-4, batch_size=256,
                 buffer_capacity=10000, target_update_freq=100,
                 rows=6, cols=7):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.rows = rows
        self.cols = cols

        # Build two networks
        self.policy_net = DQN(input_dim=43, output_dim=7)
        self.target_net = DQN(input_dim=43, output_dim=7)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.steps_done = 0

    def process_state(self, state):
        """
        state: (board, current_player)
        board shape: (6,7), current_player: float(+1/-1)
        """
        board, player = state
        board_flat = board.flatten()  # shape=(42,)
        combined = np.concatenate([board_flat, [player]])  # shape=(43,)
        combined_tensor = torch.FloatTensor(combined).unsqueeze(0)  # (1, 43)
        return combined_tensor

    def is_valid(self, board, col):
        """
        board: np.array shape=(6,7)
        Check if is valid col
        """
        if col < 0 or col >= self.cols:
            return False
        return board[0, col] == 0

    def simulate_drop(self, board, col, player):
        """
        simulate drop
        """
        sim_board = board.copy()
        drop_row = -1
        for r in range(self.rows - 1, -1, -1):
            if sim_board[r, col] == 0:
                sim_board[r, col] = player
                drop_row = r
                break
        if drop_row == -1:
            # 没找到可下的位置（不合法）
            return sim_board, False

        if self.check_win(sim_board, drop_row, col):
            return sim_board, True
        else:
            return sim_board, False

    def check_win(self, board, last_r, last_c):
        """
        Check if win
        """
        player = board[last_r, last_c]
        directions = [
            (1, 0),   # Vertical
            (0, 1),   # Horizontal
            (1, 1),   # Diagonal -> Bottom right
            (1, -1),  # Diagonal -> Bottom left
        ]
        for dr, dc in directions:
            count = 1
            # Forward
            rr, cc = last_r + dr, last_c + dc
            while 0 <= rr < self.rows and 0 <= cc < self.cols and board[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            # Backward
            rr, cc = last_r - dr, last_c - dc
            while 0 <= rr < self.rows and 0 <= cc < self.cols and board[rr, cc] == player:
                count += 1
                rr -= dr
                cc -= dc

            if count >= 4:
                return True
        return False

    def can_win_next_move(self, state):
        """
        Check if can win next move
        """
        board, player = state
        for c in range(self.cols):
            if self.is_valid(board, c):
                _, is_win = self.simulate_drop(board, c, player)
                if is_win:
                    return c
        return None

    def select_action(self, state, epsilon=0.1):  #0.1
        """
        Including win, cut off, or using neural network
        """
        board, current_player = state
        valid_cols = [c for c in range(self.cols) if self.is_valid(board, c)]

        # 1) If I can win next step
        my_winning_col = self.can_win_next_move((board, current_player))
        if my_winning_col is not None:
            return my_winning_col

        # 2) if opponent can win next step
        opponent_winning_col = self.can_win_next_move((board, -current_player))
        if opponent_winning_col is not None and opponent_winning_col in valid_cols:
            return opponent_winning_col


        # 3) Epsilon-greedy
        if np.random.rand() < epsilon:
            # Choose in valid moves
            return np.random.choice(valid_cols)
        else:
            # choose max q move
            state_tensor = self.process_state(state)  # (1,43)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)  # (7,)
            # Set invalid move q number
            masked_q = torch.full_like(q_values, float('-inf'))
            for c in valid_cols:
                masked_q[c] = q_values[c]
            action = torch.argmax(masked_q).item()
            return action

    def push_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Update nn
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Transfer to tensor
        state_tensor = torch.cat([self.process_state(s) for s in state], dim=0)     # (batch, 43)
        action_tensor = torch.LongTensor(action).unsqueeze(1)                      # (batch,1)
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1)                     # (batch,1)
        next_state_tensor = torch.cat([self.process_state(s) for s in next_state], dim=0)
        done_tensor = torch.FloatTensor(done).unsqueeze(1)                         # (batch,1)

        # Current Q
        q_values = self.policy_net(state_tensor)  # (batch,7)
        q_action = q_values.gather(1, action_tensor)  # (batch,1)

        # Double DQN: Target Q
        with torch.no_grad():
            # 1) Using policy_net choose from next_state
            next_q_values_policy = self.policy_net(next_state_tensor)       # (batch,7)
            next_actions = torch.argmax(next_q_values_policy, dim=1, keepdim=True)  # (batch,1)

            # 2) Using target_net evaluate Q
            next_q_values_target = self.target_net(next_state_tensor)  # (batch,7)
            max_next_q = next_q_values_target.gather(1, next_actions)  # (batch,1)

            q_target = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q

        loss = F.mse_loss(q_action, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        # Update target_net
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
