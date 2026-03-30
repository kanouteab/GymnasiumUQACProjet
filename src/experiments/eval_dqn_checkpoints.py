"""
Évalue les checkpoints intermédiaires du DQN pour détecter un éventuel surapprentissage.

Idée :
- charger chaque checkpoint sauvegardé en fin de phase
- mesurer son win rate contre plusieurs adversaires fixes
- comparer l'évolution des performances d'une phase à l'autre

Usage :
    python -m src.experiments.eval_dqn_checkpoints
    python -m src.experiments.eval_dqn_checkpoints --n-games 100
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from glob import glob
from typing import Dict, List, Optional, Tuple

from src.envs.othello_env import OthelloEnv, encode_action, get_winner, get_legal_moves
from src.agents.dqn import DQNAgent
from src.agents.mcts import MCTSAgent
from src.agents.alphabeta import AlphaBetaAgent

Move = Tuple[int, int]
Board = Tuple[int, int]


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        return self.rng.choice(legal) if legal else None


def evaluate(agent: DQNAgent, opp, n_games: int = 100) -> float:
    """
    Évalue le DQN (toujours Noir) contre un adversaire fixe (Blanc).
    Retourne le taux de victoire des Noirs.
    """
    saved_eps = agent.eps
    agent.eps = 0.0  # exploitation pure
    wins = 0

    for _ in range(n_games):
        env = OthelloEnv()
        env.reset()
        steps = 0

        while not env.done and steps < 200:
            if env.state.player == 1:
                obs = agent.board_to_obs(env.state.board, 1)
                legal_ids = env.legal_actions()
                a = agent.select_action(obs, legal_ids)
                env.step(a)
            else:
                mv_opp = opp.select_move(env.state.board, -1)
                env.step(encode_action(mv_opp))
            steps += 1

        if get_winner(env.state.board) == 1:
            wins += 1

    agent.eps = saved_eps
    return wins / n_games


def make_opponents(seed: int = 0) -> Dict[str, object]:
    return {
        "Random": RandomAgent(seed=seed),
        "MCTS-50": MCTSAgent(n_simulations=50, seed=seed),
        "MCTS-200": MCTSAgent(n_simulations=200, seed=seed),
        "AB-d2": AlphaBetaAgent(depth=2, use_move_ordering=True),
        "AB-d3": AlphaBetaAgent(depth=3, use_move_ordering=True),
        "AB-d4": AlphaBetaAgent(depth=4, use_move_ordering=True),
    }


def evaluate_checkpoint(
    ckpt_path: str,
    opponents: Dict[str, object],
    n_games: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Charge un checkpoint DQN et l'évalue contre tous les adversaires.
    """
    agent = DQNAgent(eps=0.0, device=device)
    agent.load(ckpt_path)

    results = {}
    for name, opp in opponents.items():
        wr = evaluate(agent, opp, n_games=n_games)
        results[name] = wr
    return results


def save_csv(rows: List[dict], out_path: str = "artifacts/dqn_checkpoint_eval.csv") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV sauvegardé → {out_path}")


def main(n_games: int = 100, device: str = "cpu") -> None:
    # Checkpoints intermédiaires + modèle final
    phase_ckpts = sorted(glob("artifacts/dqn_phase*_best.pt"))
    final_ckpt = "artifacts/dqn_model.pt"

    ckpts = []
    ckpts.extend(phase_ckpts)
    if os.path.exists(final_ckpt):
        ckpts.append(final_ckpt)

    if not ckpts:
        print("Aucun checkpoint DQN trouvé dans artifacts/.")
        return

    print("=" * 70)
    print("Évaluation des checkpoints intermédiaires du DQN")
    print(f"Nombre de parties par adversaire : {n_games}")
    print(f"Device : {device}")
    print("=" * 70)

    opponents = make_opponents(seed=42)
    rows: List[dict] = []

    for ckpt in ckpts:
        print()
        print(f"[Checkpoint] {ckpt}")
        res = evaluate_checkpoint(ckpt, opponents, n_games=n_games, device=device)

        row = {"checkpoint": os.path.basename(ckpt)}
        for opp_name, wr in res.items():
            row[opp_name] = round(wr, 4)
            print(f"  vs {opp_name:<8} : {wr:.1%}")
        rows.append(row)

    save_csv(rows)

    print()
    print("=" * 70)
    print("Analyse conseillée :")
    print("- si un checkpoint intermédiaire est meilleur que le modèle final contre certains adversaires,")
    print("  cela suggère un surapprentissage ou une spécialisation excessive en fin de curriculum ;")
    print("- si les performances montent de manière régulière, le biais observé vient moins d'un surapprentissage")
    print("  que d'une difficulté structurelle du problème ou d'un déséquilibre du curriculum.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=100, help="Parties par adversaire (défaut : 100)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu ou cuda")
    args = parser.parse_args()
    main(n_games=args.n_games, device=args.device)