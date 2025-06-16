import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToe
from model import create_model
import time

model = create_model()
game = TicTacToe()

training_episodes = 150
training_losses = []
win_rates = []

def choose_action(state, model, epsilon=0.1):
    if np.random.rand() < epsilon:
        moves = game.available_moves()
        return moves[np.random.randint(len(moves))]
    q_values = model.predict(state.reshape(1, 9), verbose=0)[0]
    for i, j in [(i, j) for i in range(3) for j in range(3) if game.board[i, j] != 0]:
        q_values[i * 3 + j] = -np.inf
    idx = np.argmax(q_values)
    return divmod(idx, 3)

# ----------- TRAINING ----------
print("\U0001F501 Tanítás elkezdődött...")
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.01

for episode in range(training_episodes):
    t0 = time.time()
    state = game.reset()
    states, actions, rewards = [], [], []
    done = False

    while not done:
        action = choose_action(state, model, epsilon)
        i, j = action
        success = game.make_move(i, j)
        if not success:
            rewards.append(-10)
            break

        next_state = game.board.flatten()
        winner = game.check_winner()

        if winner is not None:
            rewards.append(10 if winner == 1 else -10 if winner == -1 else 0)
            done = True
        else:
            rewards.append(0)

        states.append(state)
        actions.append(i * 3 + j)
        state = next_state

    total_loss = 0
    for s, a, r in zip(states, actions, rewards):
        q = model.predict(s.reshape(1, 9), verbose=0)[0]
        q[a] = r
        history = model.fit(s.reshape(1, 9), q.reshape(1, 9), epochs=1, verbose=0)
        total_loss += history.history['loss'][0]

    avg_loss = total_loss / max(1, len(states))
    training_losses.append(avg_loss)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"🧠 Epizód {episode+1}/{training_episodes} | loss: {avg_loss:.4f} | epsilon: {epsilon:.3f} | idő: {time.time() - t0:.2f}s")

# ----------- MENTÉS ÉS GÖRBE ----------
model.save("tic_tac_toe_ai.h5")
print("\n✅ Modell mentve: tic_tac_toe_ai.h5")

plt.plot(training_losses)
plt.xlabel("Epizód")
plt.ylabel("Loss")
plt.title("Tanulási görbe")
plt.grid(True)
plt.tight_layout()
plt.show()