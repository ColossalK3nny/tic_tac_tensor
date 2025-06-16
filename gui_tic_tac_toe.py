import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
from tensorflow.keras.models import load_model
from game import TicTacToe

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe AI (Tanulás játék közben)")

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
        self.model.compile(optimizer="adam", loss="mse")
        self.game = TicTacToe()

        self.memory = []  # (state, action_index, reward)
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(1, 3):
            self.canvas.create_line(0, i * 100, 300, i * 100, width=2)
            self.canvas.create_line(i * 100, 0, i * 100, 300, width=2)

        state = self.game.board.flatten()
        q_values = self.model.predict(state.reshape(1, 9), verbose=0)[0]

        q_min = np.min(q_values)
        q_max = np.max(q_values)
        q_range = q_max - q_min + 1e-6
        q_normalized = (q_values - q_min) / q_range

        for i in range(3):
            for j in range(3):
                x0, y0 = j * 100, i * 100
                x1, y1 = x0 + 100, y0 + 100
                idx = i * 3 + j
                if self.game.board[i, j] == 0:
                    green = int(q_normalized[idx] * 255)
                    color = f"#{255-green:02x}{255:02x}{255-green:02x}"
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                    self.canvas.create_text(x0 + 50, y0 + 80, text=f"{q_values[idx]:.2f}", font=("Helvetica", 10), fill="black")

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

        self.memory.append((state, best_move))

        self.game.make_move(row, col)
        self.draw_board()

        winner = self.game.check_winner()
        if winner is not None:
            self.end_game(winner)

        print(f"🤖 AI confidence: {np.round(q_values, 2)}")

    def end_game(self, winner):
        if winner == 1:
            text = "Te nyertél!"
        elif winner == -1:
            text = "AI nyert!"
        else:
            text = "Döntetlen!"
        self.status_label.config(text=text)

        if self.memory:
            for state, action_index in self.memory:
                q = self.model.predict(state.reshape(1, 9), verbose=0)[0]
                if winner == -1:
                    q[action_index] = 1
                elif winner == 1:
                    q[action_index] = -1
                else:
                    q[action_index] = 0
                self.model.fit(state.reshape(1, 9), q.reshape(1, 9), epochs=1, verbose=0)
            self.model.save("tic_tac_toe_ai.h5")
            print("🧠 Modell frissítve és mentve tanulás után.")

    def reset_game(self):
        self.game.reset()
        self.memory = []
        self.status_label.config(text="Te kezded!")
        self.draw_board()

if __name__ == "__main__":
    root = ttk.Window(themename="superhero")
    app = TicTacToeGUI(root)
    root.mainloop()