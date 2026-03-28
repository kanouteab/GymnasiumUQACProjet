"""
run_all.py — Point d'entrée unique du projet GymnasiumUQACProjet.

Lance dans l'ordre :
  1. Entraînement Q-Learning (12 000 épisodes, curriculum 5 phases)
  2. Entraînement DQN        (5 000 épisodes par défaut, CNN 3 canaux)
  3. Tournoi inter-agents    (Random, MCTS, AlphaBeta-d2/d3/d4, QL, DQN)

Usage :
    python run_all.py

Options :
    --skip-train       Ignorer l'entraînement Q-Learning
    --skip-dqn         Ignorer l'entraînement DQN
    --skip-tournament  Ignorer le tournoi
    --fresh            Supprimer artifacts/qtable.pkl et dqn_model.pt avant de commencer
    --tournament-games Nombre de parties par matchup (défaut : 20)
    --dqn-episodes     Nombre d'épisodes DQN (défaut : 20 000)
"""
from __future__ import annotations

import argparse
import os
import sys


def _separator(title: str) -> None:
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def step_train() -> None:
    _separator("Étape 1 — Entraînement Q-Learning")
    from src.experiments.train_rl import main as train
    train()


def step_train_dqn(n_episodes: int = 5_000) -> None:
    _separator("Étape 2 — Entraînement DQN (CNN)")
    from src.experiments.train_dqn import main as train_dqn
    train_dqn(n_episodes=n_episodes)


def step_tournament(n_games: int = 20, static_only: bool = False) -> None:
    _separator("Étape 3 — Tournoi inter-agents")
    from src.experiments.tournament import main as tournament_main
    tournament_main(n_games=n_games, static_only=static_only)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance tout le pipeline du projet.")
    parser.add_argument("--skip-train",      action="store_true", help="Ignorer l'entraînement Q-Learning")
    parser.add_argument("--skip-dqn",        action="store_true", help="Ignorer l'entraînement DQN")
    parser.add_argument("--skip-tournament", action="store_true", help="Ignorer le tournoi inter-agents")
    parser.add_argument("--fresh",           action="store_true", help="Supprimer qtable.pkl et dqn_model.pt avant de commencer")
    parser.add_argument("--tournament-games", type=int, default=20, help="Parties par matchup (défaut : 20)")
    parser.add_argument("--dqn-episodes",    type=int, default=20_000, help="Épisodes DQN (défaut : 20 000)")
    parser.add_argument("--static-tournament", action="store_true",
                        help="Tournoi agents statiques seulement (sans QL/DQN) "
                             "pour vérifier l'ordre de force du curriculum")
    args = parser.parse_args()

    if args.fresh:
        for path in ("artifacts/qtable.pkl", "artifacts/dqn_model.pt"):
            if os.path.exists(path):
                os.remove(path)
                print(f"{path} supprimé.")

    if not args.skip_train:
        step_train()
    else:
        print("\n[skip] Entraînement Q-Learning ignoré.")
        if not os.path.exists("artifacts/qtable.pkl"):
            print("ERREUR : artifacts/qtable.pkl introuvable et --skip-train activé.")
            sys.exit(1)

    if not args.skip_dqn:
        step_train_dqn(n_episodes=args.dqn_episodes)
    else:
        print("\n[skip] Entraînement DQN ignoré.")
        if not os.path.exists("artifacts/dqn_model.pt"):
            print("  AVERTISSEMENT : dqn_model.pt introuvable — DQN jouera non entraîné dans le tournoi.")

    if not args.skip_tournament:
        step_tournament(n_games=args.tournament_games, static_only=args.static_tournament)
    else:
        print("\n[skip] Tournoi ignoré.")

    print()
    print("=" * 65)
    print("  Terminé.")
    print("=" * 65)


if __name__ == "__main__":
    main()
