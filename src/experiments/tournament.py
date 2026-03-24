# src/experiments/tournament.py
"""
Tournoi complet entre tous les agents.

Agents :
  - Random
  - MCTS (200 simulations)
  - AlphaBeta depth=2, 3, 4
  - Q-Learning (chargé depuis artifacts/qtable.pkl)

Pour chaque paire (i, j) avec i≠j :
  → n_games parties, agent i joue les Noirs, agent j les Blancs.
  → matrix[i][j] = taux de victoire de i (Noirs).

Résultats sauvegardés dans :
  artifacts/tournament.csv   — données brutes
  artifacts/tournament.png   — heatmap (via plot_results.py)

Lancement direct :
    conda run -n uqac-gymnasium python -m src.experiments.tournament
    conda run -n uqac-gymnasium python -m src.experiments.tournament --n-games 50
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

from src.envs.othello_env import (
    apply_move, get_legal_moves, get_winner, initial_board, is_terminal,
)
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.mcts import MCTSAgent
from src.agents.qlearning import QLearningAgent
from src.training.features import action_to_id, id_to_action, state_features

Move  = Tuple[int, int]
Board = Tuple[int, int]


# ── Agents utilitaires ─────────────────────────────────────────────────────────

class _RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        return self.rng.choice(legal) if legal else None


class _QLWrapper:
    """Enveloppe QLearningAgent avec l'interface select_move()."""

    def __init__(self, agent: QLearningAgent) -> None:
        self._agent = agent

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        s    = state_features(board, player)
        aids = [action_to_id(mv) for mv in legal]
        a    = self._agent.best_action(s, aids)
        return id_to_action(a)


# ── Simulation ─────────────────────────────────────────────────────────────────

def _play_one(black, white, max_steps: int = 200) -> int:
    """Joue une partie complète. Retourne +1 (Noirs), -1 (Blancs) ou 0 (nul)."""
    board = initial_board()
    steps = 0
    while not is_terminal(board) and steps < max_steps:
        mv = black.select_move(board, 1)
        if mv is not None:
            board = apply_move(board, 1, mv)
        steps += 1
        if is_terminal(board):
            break
        mv = white.select_move(board, -1)
        if mv is not None:
            board = apply_move(board, -1, mv)
        steps += 1
    return get_winner(board)


def _play_n(
    black,
    white,
    n_games: int,
) -> Tuple[int, int, int]:
    """
    Joue n_games parties (black=Noirs, white=Blancs).
    Réinitialise l'arbre MCTS entre chaque partie si la méthode existe.
    Retourne (victoires_noirs, victoires_blancs, nuls).
    """
    wb = ww = d = 0
    for _ in range(n_games):
        if hasattr(black, "reset_tree"):
            black.reset_tree()
        if hasattr(white, "reset_tree"):
            white.reset_tree()
        w = _play_one(black, white)
        if w == 1:
            wb += 1
        elif w == -1:
            ww += 1
        else:
            d += 1
    return wb, ww, d


# ── Worker pour le tournoi parallèle ─────────────────────────────────────────

# Agents créés une fois par processus worker (via _init_pool)
_WORKER_AGENTS: list = []


def _init_pool(qtable_path: str, seed: int) -> None:
    """Initialise les agents une fois par processus worker."""
    global _WORKER_AGENTS
    ql_agent = QLearningAgent(eps=0.0)
    if os.path.exists(qtable_path):
        with open(qtable_path, "rb") as f:
            data = pickle.load(f)
            ql_agent.Q1 = data["Q1"]
            ql_agent.Q2 = data["Q2"]
    _WORKER_AGENTS = [
        _RandomAgent(seed=seed),
        MCTSAgent(n_simulations=200, seed=seed),
        AlphaBetaAgent(depth=2, use_move_ordering=True),
        AlphaBetaAgent(depth=3, use_move_ordering=True),
        AlphaBetaAgent(depth=4, use_move_ordering=True),
        _QLWrapper(ql_agent),
    ]


def _run_matchup(args: tuple) -> tuple:
    """Joue un matchup (i, j) dans le processus worker courant."""
    i, j, n_games = args
    wb, ww, d = _play_n(_WORKER_AGENTS[i], _WORKER_AGENTS[j], n_games=n_games)
    return i, j, wb, ww, d


# ── Tournoi ────────────────────────────────────────────────────────────────────

def run_tournament(
    qtable_path: str = "artifacts/qtable.pkl",
    n_games:     int = 20,
    seed:        int = 0,
) -> Tuple[List[List[float]], List[str]]:
    """
    Fait jouer chaque paire d'agents n_games parties en parallèle.
    matrix[i][j] = taux de victoire de l'agent i (Noirs) contre l'agent j (Blancs).
    La diagonale vaut 0.5 par convention.
    """
    names = ["Random", "MCTS-200", "AB-d2", "AB-d3", "AB-d4", "QL"]
    n     = len(names)
    matrix: List[List[float]] = [[0.5] * n for _ in range(n)]

    tasks  = [(i, j, n_games) for i in range(n) for j in range(n) if i != j]
    total  = len(tasks)
    n_cpu  = os.cpu_count() or 4
    # Réserver 1 cœur pour le processus principal
    n_workers = max(1, n_cpu - 1)

    print(f"  {total} matchups — {n_workers} processus parallèles (sur {n_cpu} cœurs)")

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_pool,
        initargs=(qtable_path, seed),
    ) as executor:
        future_map = {executor.submit(_run_matchup, t): t for t in tasks}
        for future in as_completed(future_map):
            i, j, wb, ww, d = future.result()
            matrix[i][j] = wb / n_games
            done_count += 1
            print(
                f"  [{done_count:2d}/{total}] {names[i]:<10} (N) vs {names[j]:<10} (B)"
                f" — N {wb/n_games:.0%}  B {ww/n_games:.0%}  nul {d/n_games:.0%}"
            )

    return matrix, names


# ── Sauvegarde CSV ─────────────────────────────────────────────────────────────

def save_csv(
    matrix:      List[List[float]],
    agent_names: List[str],
    out_path:    str = "artifacts/tournament.csv",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Agent_Noir \\ Agent_Blanc"] + agent_names)
        for name, row in zip(agent_names, matrix):
            writer.writerow([name] + [f"{v:.4f}" for v in row])
    print(f"  Résultats CSV → {out_path}")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main(n_games: int = 20) -> None:
    from src.experiments.plot_results import plot_tournament

    print("=" * 65)
    print("  Tournoi inter-agents")
    print(f"  {n_games} parties par sens de jeu (N vs B)")
    print("=" * 65)

    matrix, names = run_tournament(n_games=n_games)
    save_csv(matrix, names)

    print()
    _print_summary(matrix, names)

    plot_tournament(matrix, names)


def _print_summary(matrix: List[List[float]], names: List[str]) -> None:
    """Affiche un résumé textuel dans le terminal."""
    import numpy as np
    mat = np.array(matrix)
    n   = len(names)

    # Win rate moyen en Noirs (sans diagonale)
    mask = ~np.eye(n, dtype=bool)
    avg_black = np.where(mask, mat, np.nan)
    wr_black  = np.nanmean(avg_black, axis=1)

    # Win rate moyen en Blancs = 1 - win rate de l'adversaire en Noirs
    mat_as_white = 1.0 - mat.T
    np.fill_diagonal(mat_as_white, np.nan)
    wr_white = np.nanmean(mat_as_white, axis=1)

    # Moyenne globale
    wr_global = (wr_black + wr_white) / 2.0

    ranking = sorted(zip(names, wr_black, wr_white, wr_global),
                     key=lambda x: x[3], reverse=True)

    print(f"  {'Agent':<12} {'Moy. Noirs':>11} {'Moy. Blancs':>12} {'Moy. glob.':>11}")
    print("  " + "-" * 48)
    for rank, (name, wb, ww, wg) in enumerate(ranking, 1):
        print(f"  {rank}. {name:<10} {wb:>10.1%} {ww:>12.1%} {wg:>11.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=20,
                        help="Nombre de parties par matchup (défaut : 20)")
    args = parser.parse_args()
    main(n_games=args.n_games)
