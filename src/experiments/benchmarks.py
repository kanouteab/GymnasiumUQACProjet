# src/experiments/benchmarks.py
import os
import random
from concurrent.futures import ProcessPoolExecutor
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


# ─────────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────────

class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select_move(self, board, player):
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        return self.rng.choice(legal)


def _make_agent(spec: dict, game_seed: int = 0):
    """Reconstruit un agent depuis un dict de paramètres (utilisé dans les workers)."""
    kind = spec["kind"]
    if kind == "mcts":
        return MCTSAgent(n_simulations=spec["n_simulations"], c_uct=spec["c_uct"], seed=game_seed)
    if kind == "alphabeta":
        return AlphaBetaAgent(depth=spec["depth"], use_move_ordering=spec["use_move_ordering"])
    if kind == "random":
        return RandomAgent(seed=game_seed)  # seed de partie = parties distinctes
    raise ValueError(f"Agent inconnu : {kind}")


# ─────────────────────────────────────────────────────────────────
# Logique de partie
# ─────────────────────────────────────────────────────────────────

def play_game(agent_black, agent_white, seed=0, verbose=False):
    rng = random.Random(seed)
    board = initial_board()
    player = 1  # noir commence

    while not is_terminal(board):
        legal = get_legal_moves(board, player)

        move = agent_black.select_move(board, player) if player == 1 \
               else agent_white.select_move(board, player)

        # Sécurité : si l'agent renvoie un coup invalide, coup aléatoire
        if move is not None:
            if move not in legal:
                move = rng.choice(legal) if legal else None
            if move is not None:
                board = apply_move(board, player, move)

        player = -player

        if verbose:
            print("Score:", score(board))

    return get_winner(board), score(board)


def _worker(args):
    """
    Fonction top-level exécutée dans un processus fils.
    Doit être top-level (pas lambda, pas méthode) pour être picklable sur Windows.

    Sur Windows, spawn recrée le processus from scratch → Numba charge le cache
    compilé (fichiers .nbc/.nbi dans __pycache__) pour chaque worker.
    Les agents sont reconstruits localement : aucun état partagé entre processus.
    Le game_seed est utilisé pour RandomAgent et MCTS afin que chaque partie
    soit différente (évite 20 parties identiques avec le même seed fixe).
    """
    black_spec, white_spec, seed = args
    black = _make_agent(black_spec, game_seed=seed)
    white = _make_agent(white_spec, game_seed=seed)
    return play_game(black, white, seed=seed)


# ─────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────

def bench(n_games=20, seed=0, mcts_sims=800, ab_depth=4):
    rng = random.Random(seed)

    # Specs sérialisables (dict) plutôt qu'objets agents
    mcts_spec = {"kind": "mcts",      "n_simulations": mcts_sims, "c_uct": 1.4, "seed": seed}
    ab_spec   = {"kind": "alphabeta", "depth": ab_depth, "use_move_ordering": True}
    rnd_spec  = {"kind": "random",    "seed": seed}

    matchups = [
        ("MCTS(Black) vs Random(White)",    mcts_spec, rnd_spec),
        ("AlphaBeta(Black) vs Random(White)", ab_spec,  rnd_spec),
        ("MCTS(Black) vs AlphaBeta(White)", mcts_spec, ab_spec),
    ]

    # Nombre de workers : tous les cœurs logiques du CPU
    n_workers = os.cpu_count() or 1

    for title, black_spec, white_spec in matchups:
        seeds = [rng.randint(0, 10_000) for _ in range(n_games)]
        args_list = [(black_spec, white_spec, s) for s in seeds]

        wins = {1: 0, -1: 0, 0: 0}
        t0 = perf_counter()

        # Chaque partie s'exécute dans un processus fils indépendant.
        # ProcessPoolExecutor distribue les n_games tâches sur n_workers processus.
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for winner, _ in executor.map(_worker, args_list):
                wins[winner] += 1

        t1 = perf_counter()
        print("=" * 60)
        print(title)
        print(f"Games: {n_games} | Workers: {n_workers} | Time: {round(t1 - t0, 2)} s")
        print(f"Black wins: {wins[1]} | White wins: {wins[-1]} | Draw: {wins[0]}")

if __name__ == "__main__":
    bench(n_games=20, seed=0, mcts_sims=800, ab_depth=4)
