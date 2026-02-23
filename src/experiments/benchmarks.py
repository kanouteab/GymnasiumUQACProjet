# src/experiments/benchmarks.py
import random
from time import perf_counter

from src.envs.othello_env import (
    initial_board,
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
)
from src.agents.mcts import MCTSAgent
from src.agents.alphabeta import AlphaBetaAgent


def play_game(agent_black, agent_white, seed=0, verbose=False):
    rng = random.Random(seed)
    board = initial_board()
    player = 1  # noir

    while not is_terminal(board):
        legal = get_legal_moves(board, player)

        if player == 1:
            move = agent_black.select_move(board, player)
        else:
            move = agent_white.select_move(board, player)

        # Si agent renvoie None => PASS (doit être légal si aucun move)
        if move is not None:
            if move not in legal:
                # en cas de bug: forcer un coup aléatoire légal
                move = rng.choice(legal) if legal else None
            if move is not None:
                board = apply_move(board, player, move)

        player = -player

        if verbose:
            print("Score:", score(board))

    return get_winner(board), score(board)


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_move(self, board, player):
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        return self.rng.choice(legal)


def bench(n_games=20, seed=0, mcts_sims=800, ab_depth=4):
    rng = random.Random(seed)

    mcts = MCTSAgent(n_simulations=mcts_sims, c_uct=1.4, seed=seed)
    ab = AlphaBetaAgent(depth=ab_depth, use_move_ordering=True)
    rnd = RandomAgent(seed=seed)

    matchups = [
        ("MCTS(Black) vs Random(White)", mcts, rnd),
        ("AlphaBeta(Black) vs Random(White)", ab, rnd),
        ("MCTS(Black) vs AlphaBeta(White)", mcts, ab),
    ]

    for title, black, white in matchups:
        wins = {1: 0, -1: 0, 0: 0}
        t0 = perf_counter()
        for i in range(n_games):
            w, s = play_game(black, white, seed=rng.randint(0, 10_000))
            wins[w] += 1
        t1 = perf_counter()
        print("=" * 60)
        print(title)
        print("Games:", n_games, "| Time:", round(t1 - t0, 2), "s")
        print("Black wins:", wins[1], "| White wins:", wins[-1], "| Draw:", wins[0])


if __name__ == "__main__":
    bench(n_games=20, seed=0, mcts_sims=800, ab_depth=4)
