
# src/experiments/run_mcts_vs_random.py
import random
import numpy as np

from src.envs.othello_env import initial_board, get_legal_moves, apply_move, is_terminal, get_winner, score
from src.agents.mcts import MCTSAgent


def play_one_game(mcts_sims=800, seed=0, mcts_player=1, verbose=False):
    rng = random.Random(seed)
    board = initial_board()
    player = 1  # noir commence

    agent = MCTSAgent(n_simulations=mcts_sims, c_uct=1.4, seed=seed)

    while not is_terminal(board):
        legal = get_legal_moves(board, player)

        if player == mcts_player:
            move = agent.select_move(board, player)  # may be None (PASS)
        else:
            move = rng.choice(legal) if legal else None

        if move is not None:
            board = apply_move(board, player, move)
        # else PASS
        player = -player

        if verbose:
            print("Player:", "N" if player == 1 else "B", "Score:", score(board))

    w = get_winner(board)
    if verbose:
        print("Final score:", score(board), "Winner:", w)
    return w, score(board)


def main():
    wins = {1: 0, -1: 0, 0: 0}
    n_games = 10
    for i in range(n_games):
        w, s = play_one_game(mcts_sims=800, seed=i, mcts_player=1, verbose=False)
        wins[w] += 1

    print("MCTS (Noir) vs Random (Blanc) sur", n_games, "parties")
    print("Noir gagne:", wins[1], "| Blanc gagne:", wins[-1], "| Nul:", wins[0])


if __name__ == "__main__":
    main()
