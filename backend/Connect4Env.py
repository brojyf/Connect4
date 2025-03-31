import numpy as np

class Connect4Env:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = None  # Board: shape = (rows, cols)
        self.current_player = None  # Current player：+1 (X) || -1 (O)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_player = 1.0  # First player is +1
        # return：(board, current_player)
        return (self.board, self.current_player)

    def is_valid_action(self, col):
        """Check if is valid action"""
        if col < 0 or col >= self.cols:
            return False
        # if row 0 is not empty, invalid action
        return self.board[0, col] == 0

    def step(self, action):
        """
        Execute one step of the game, action represent the col
        return: (next_state, reward, done, info)
        """
        info = {"invalid": False}

        # If  invalid move
        if not self.is_valid_action(action):
            done = True
            reward = -1.0  # Punish invalid move
            info["invalid"] = True
            return (self.board, self.current_player), reward, done, info

        # Put in the correct position
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        # Check if game is over
        done, winner = self.check_done()
        if winner == self.current_player:
            # Current player wins
            reward = 1.0
        elif winner is None and done:
            # Tie
            reward = 0.0
        else:
            # Keep playing
            reward = 0.0

        # If game is not over, keep playing
        if not done:
            self.current_player = -self.current_player

        return (self.board, self.current_player), reward, done, info

    def check_done(self):
        """
        check if one player wins or the board is full, return (done, winner)
        winner = +1 || -1 represents the winner，None means a tie
        """
        board = self.board
        rows, cols = self.rows, self.cols

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                line = board[r, c:c+4]
                if abs(line.sum()) == 4 and len(set(line)) == 1:
                    return True, line[0]

        # vertical
        for c in range(cols):
            for r in range(rows - 3):
                line = board[r:r+4, c]
                if abs(line.sum()) == 4 and len(set(line)) == 1:
                    return True, line[0]

        # diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                line = [board[r + i, c + i] for i in range(4)]
                if abs(sum(line)) == 4 and len(set(line)) == 1:
                    return True, line[0]

        # diagonal
        for r in range(rows - 3):
            for c in range(3, cols):
                line = [board[r + i, c - i] for i in range(4)]
                if abs(sum(line)) == 4 and len(set(line)) == 1:
                    return True, line[0]

        # check is if full
        if np.all(board != 0):
            return True, None  # Tie

        return False, None

    def render(self):
        """Print board"""
        print("\nCurrent Board:")
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                val = self.board[r, c]
                if val == 1:
                    row_str += "X "
                elif val == -1:
                    row_str += "O "
                else:
                    row_str += ". "
            print(row_str)
        print("-------------")
        print(" ".join(str(c) for c in range(self.cols)))
        print()
