import numpy as np
from tensorflow.keras.models import load_model
from game import TicTacToe

def generate_test_data(n=100):
    game = TicTacToe()
    data = []
    for _ in range(n):
        game.reset()
        done = False
        while not done:
            state = game.board.flatten()
            moves = game.available_moves()
            i, j = moves[np.random.randint(len(moves))]
            success = game.make_move(i, j)
            if not success:
                break
            winner = game.check_winner()
            reward = 1 if winner == -1 else -1 if winner == 1 else 0
            if winner is not None:
                done = True
            action_index = i * 3 + j
            data.append((state, action_index, reward))
    return data

def evaluate_model(model, data):
    total_loss = 0
    for state, action_index, reward in data:
        q_pred = model.predict(state.reshape(1, 9), verbose=0)[0]
        target = np.copy(q_pred)
        target[action_index] = reward
        loss = np.mean((q_pred - target) ** 2)
        total_loss += loss
    return total_loss / len(data)

test_data = generate_test_data()

before = load_model("tic_tac_toe_ai_before.h5", compile=False)
before.compile(optimizer="adam", loss="mse")
after = load_model("tic_tac_toe_ai.h5", compile=False)
after.compile(optimizer="adam", loss="mse")

before_loss = evaluate_model(before, test_data)
after_loss = evaluate_model(after, test_data)

print(f"📉 Loss előtte: {before_loss:.4f}")
print(f"📈 Loss utána:  {after_loss:.4f}")
print("🎯 Fejlődés:", "igen" if after_loss < before_loss else "nem")