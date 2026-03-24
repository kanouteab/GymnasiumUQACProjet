"""
run_all.py — Point d'entrée unique du projet GymnasiumUQACProjet.

Lance dans l'ordre :
  1. Entraînement Q-Learning (12 000 épisodes, curriculum 5 phases)
  2. Tournoi inter-agents (Random, MCTS, AlphaBeta-d2/d3/d4, QL)

Usage :
    conda run -n uqac-gymnasium python run_all.py

Options :
    --skip-train       Ignorer l'entraînement (utile si artifacts/qtable.pkl existe déjà)
    --skip-tournament  Ignorer le tournoi
    --fresh            Supprimer artifacts/qtable.pkl avant de commencer
    --tournament-games Nombre de parties par matchup (défaut : 20)
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


def step_tournament(n_games: int = 20) -> None:
    _separator("Étape 2 — Tournoi inter-agents")
    from src.experiments.tournament import main as tournament_main
    tournament_main(n_games=n_games)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance tout le pipeline du projet.")
    parser.add_argument("--skip-train",      action="store_true", help="Ignorer l'entraînement RL")
    parser.add_argument("--skip-tournament", action="store_true", help="Ignorer le tournoi inter-agents")
    parser.add_argument("--fresh",           action="store_true", help="Supprimer qtable.pkl avant de commencer")
    parser.add_argument("--tournament-games", type=int, default=20, help="Parties par matchup (défaut : 20)")
    args = parser.parse_args()

    if args.fresh and os.path.exists("artifacts/qtable.pkl"):
        os.remove("artifacts/qtable.pkl")
        print("artifacts/qtable.pkl supprimé.")

    if not args.skip_train:
        step_train()
    else:
        print("\n[skip] Entraînement ignoré.")
        if not os.path.exists("artifacts/qtable.pkl"):
            print("ERREUR : artifacts/qtable.pkl introuvable et --skip-train activé.")
            sys.exit(1)

    if not args.skip_tournament:
        step_tournament(n_games=args.tournament_games)
    else:
        print("\n[skip] Tournoi ignoré.")

    print()
    print("=" * 65)
    print("  Terminé.")
    print("=" * 65)


if __name__ == "__main__":
    main()
