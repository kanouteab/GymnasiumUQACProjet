# src/experiments/train_rl.py
"""
Command to launch training:
python -m src.experiments.train_rl
"""
from __future__ import annotations
import random
from typing import Optional, Tuple, List

from src.envs.othello_env import (
    initial_board, get_legal_moves, apply_move,
    is_terminal, get_winner
)
from src.rl.features import state_features, action_to_id, id_to_action
from src.rl.qlearning import QLearningAgent

Move = Tuple[int, int]
Board = Tuple[int, int]  # (black_bb, white_bb)

class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        return self.rng.choice(legal)

def legal_action_ids(board: Board, player: int) -> List[int]:
    legal = get_legal_moves(board, player)
    if not legal:
        return [-1]  # PASS
    return [action_to_id(mv) for mv in legal]

def play_episode_train(agent: QLearningAgent, opp: RandomAgent, seed: int = 0, max_steps: int = 200):
    rng = random.Random(seed)
    board = initial_board()
    player = 1  # black starts

    # on entraîne RL uniquement quand player==1 (noir)
    steps = 0
    while not is_terminal(board) and steps < max_steps:
        if player == 1:
            s = state_features(board, player)
            legal_a = legal_action_ids(board, player)
            a = agent.select_action(s, legal_a)
            mv = id_to_action(a)

            # appliquer coup (ou PASS)
            if mv is not None:
                board = apply_move(board, player, mv)
            player = -player

            # adversaire joue 1 coup (ou PASS)
            if not is_terminal(board):
                mv2 = opp.select_move(board, player)
                if mv2 is not None:
                    board = apply_move(board, player, mv2)
                player = -player

            done = is_terminal(board) or steps >= (max_steps - 1)

            # reward seulement à la fin (simple)
            if done:
                w = get_winner(board)  # +1/-1/0
                r = 1.0 if w == 1 else (-1.0 if w == -1 else 0.0)
                s2 = state_features(board, 1)  # état final vu par noir
                legal_a2 = legal_action_ids(board, 1)
                agent.update(s, a, r, s2, legal_a2, done=True)
            else:
                # reward intermédiaire 0
                s2 = state_features(board, 1)
                legal_a2 = legal_action_ids(board, 1)
                agent.update(s, a, 0.0, s2, legal_a2, done=False)

        else:
            # (normalement jamais ici, car on fait jouer l'adversaire immédiatement)
            mv2 = opp.select_move(board, player)
            if mv2 is not None:
                board = apply_move(board, player, mv2)
            player = -player

        steps += 1

    agent.decay_epsilon()
    w = get_winner(board)
    return w

def main():
    agent = QLearningAgent(alpha=0.2, gamma=0.95, eps=1.0, eps_decay=0.995, eps_min=0.05, seed=0)
    opp = RandomAgent(seed=1)

    n_episodes = 2000  # augmenter si on veut plus d'episodes
    wins = {1:0, -1:0, 0:0}
    for ep in range(1, n_episodes+1):
        w = play_episode_train(agent, opp, seed=ep)
        wins[w] += 1

        if ep % 100 == 0:
            print(f"Episode {ep} | eps={agent.eps:.3f} | wins(B)={wins[1]} losses={wins[-1]} draws={wins[0]}")

    # sauvegarde simple (pickle)
    import pickle, os
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/qtable.pkl", "wb") as f:
        pickle.dump(agent.Q, f)
    print("Saved Q-table -> artifacts/qtable.pkl")

if __name__ == "__main__":

    main()
