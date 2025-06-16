import numpy as np

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 = X, -1 = O
        return self.board.flatten()

    def available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, i, j):
        if self.board[i, j] != 0:
            return False
        self.board[i, j] = self.current_player
        self.current_player *= -1
        return True

    def check_winner(self):
        lines = list(self.board) + list(self.board.T) + \
                [self.board.diagonal(), np.fliplr(self.board).diagonal()]
        for line in lines:
            s = sum(line)
            if s == 3:
                return 1
            elif s == -3:
                return -1
        if np.all(self.board != 0):
            return 0  # draw
        return None