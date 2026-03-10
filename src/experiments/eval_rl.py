# src/experiments/eval_rl.py
from __future__ import annotations
import random
import pickle
from typing import Optional, Tuple

from src.envs.othello_env import (
    initial_board, get_legal_moves, apply_move,
    is_terminal, get_winner, score
)
from src.rl.features import state_features, action_to_id, id_to_action
from src.envs.othello_env import get_legal_moves
from src.rl.qlearning import QLearningAgent

def legal_action_ids(board, player):
    legal = get_legal_moves(board, player)
    if not legal:
        return [-1]
    return [action_to_id(m) for m in legal]
Move = Tuple[int, int]
Board = Tuple[int, int]

class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        return self.rng.choice(legal)

def play_game(agent_black: QLearningAgent, agent_white: RandomAgent, seed=0):
    rng = random.Random(seed)
    board = initial_board()
    player = 1

    while not is_terminal(board):
        if player == 1:
            s = state_features(board, 1)
            legal = get_legal_moves(board, 1)
            if not legal:
                mv = None
            else:
                legal_aids = [action_to_id(m) for m in legal]
                a = agent_black.best_action(s, legal_aids)  # greedy en eval
                mv = id_to_action(a)
        else:
            mv = agent_white.select_move(board, -1)

        if mv is not None:
            board = apply_move(board, player, mv)
        player = -player

    return get_winner(board), score(board)

def main():
    agent = QLearningAgent(eps=0.0)  # greedy
    with open("artifacts/qtable.pkl", "rb") as f:
        agent.Q = pickle.load(f)

    opp = RandomAgent(seed=1)

    n_games = 100
    wins = {1:0, -1:0, 0:0}
    for i in range(n_games):
        w, s = play_game(agent, opp, seed=i)
        wins[w] += 1
    print("RL(Black) vs Random(White)")
    print("Games:", n_games, "| Black wins:", wins[1], "| White wins:", wins[-1], "| Draw:", wins[0])

if __name__ == "__main__":
    main()