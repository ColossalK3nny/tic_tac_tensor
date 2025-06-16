
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
from tensorflow.keras.models import load_model
from game import TicTacToe

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe AI (TensorFlow)")

        self.frame = ttk.Frame(root, padding=10)
        self.frame.pack()

        self.canvas = ttk.Canvas(self.frame, width=300, height=300)
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.canvas.bind("<Button-1>", self.click)

        self.status_label = ttk.Label(self.frame, text="Te kezded!", font=("Helvetica", 14))
        self.status_label.grid(row=1, column=0, pady=10)

        self.reset_button = ttk.Button(self.frame, text="Új játék", command=self.reset_game, bootstyle="success")
        self.reset_button.grid(row=1, column=1, pady=10)

        self.model = load_model("tic_tac_toe_ai.h5", compile=False)
        self.game = TicTacToe()

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(1, 3):
            self.canvas.create_line(0, i * 100, 300, i * 100, width=2)
            self.canvas.create_line(i * 100, 0, i * 100, 300, width=2)

        for i in range(3):
            for j in range(3):
                x = j * 100 + 50
                y = i * 100 + 50
                if self.game.board[i, j] == 1:
                    self.canvas.create_text(x, y, text="X", font=("Helvetica", 48))
                elif self.game.board[i, j] == -1:
                    self.canvas.create_text(x, y, text="O", font=("Helvetica", 48))

    def click(self, event):
        if self.game.check_winner() is not None:
            self.status_label.config(text="Játék vége! Kattints az Új játék gombra.")
            return

        row = event.y // 100
        col = event.x // 100

        if self.game.board[row, col] != 0:
            return

        self.game.make_move(row, col)
        self.draw_board()

        winner = self.game.check_winner()
        if winner is not None:
            self.end_game(winner)
            return

        self.root.after(500, self.ai_move)

    def ai_move(self):
        state = self.game.board.flatten()
        q_values = self.model.predict(state.reshape(1, 9), verbose=0)[0]

        for i, j in [(i, j) for i in range(3) for j in range(3) if self.game.board[i, j] != 0]:
            q_values[i * 3 + j] = -np.inf

        best_move = np.argmax(q_values)
        row, col = divmod(best_move, 3)

        self.game.make_move(row, col)
        self.draw_board()

        winner = self.game.check_winner()
        if winner is not None:
            self.end_game(winner)

    def end_game(self, winner):
        if winner == 1:
            text = "Te nyertél!"
        elif winner == -1:
            text = "AI nyert!"
        else:
