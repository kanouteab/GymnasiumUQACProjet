# src/experiments/train_dqn.py
"""
Entraînement DQN pour Othello.

Architecture : CNN 3 canaux (mes pions / pions adverses / coups légaux)
Device       : CUDA si disponible, sinon CPU
Curriculum   :
  Phase 1 (ep    1–1 000) : vs Random         — apprentissage des bases
  Phase 2 (ep 1 001–2 000) : vs MCTS-200      — stratégie positionnelle
  Phase 3 (ep 2 001–3 000) : vs AlphaBeta-d2  — vision tactique
  Phase 4 (ep 3 001–4 000) : vs AlphaBeta-d3  — affinage fort
  Phase 5 (ep 4 001–5 000) : vs AlphaBeta-d4  — test ultime

Boucle par épisode :
  - L'agent joue toujours les Noirs (+1)
  - OthelloEnv (interface Gymnasium) pilote la boucle de jeu
  - Après chaque (coup agent + réponse adversaire) → push() + update()
  - Reward shaping : coins ±0.3, mobilité ×0.01 (petits vs ±1 terminal)

Sorties :
  artifacts/dqn_model.pt              — poids du réseau
  artifacts/dqn_training_stats.csv    — win_rate / eps / avg_loss par tranche
  artifacts/dqn_learning_curve.png    — courbe 3 panneaux (win rate, loss, ε)
  artifacts/dqn_final_eval.png        — barres d'évaluation finale

Lancement :
    python -m src.experiments.train_dqn
    python -m src.experiments.train_dqn --episodes 10000
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import List, Optional, Tuple

import torch

from src.envs.othello_env import (
    OthelloEnv, encode_action, get_winner,
)
from src.agents.dqn import DQNAgent
from src.agents.mcts import MCTSAgent
from src.agents.alphabeta import AlphaBetaAgent
from src.training.features import shaped_reward

Board = Tuple[int, int]
Move  = Tuple[int, int]


# ── Agent adversaire Random ────────────────────────────────────────────────────

class _RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        from src.envs.othello_env import get_legal_moves
        self._get_legal = get_legal_moves
        self.rng = random.Random(seed)

    def select_move(self, board, player) -> Optional[Move]:
        legal = self._get_legal(board, player)
        return self.rng.choice(legal) if legal else None


# ── Boucle de jeu (1 épisode) ─────────────────────────────────────────────────

def play_episode(
    agent:      DQNAgent,
    opp,
    shaped:     bool = True,
    max_steps:  int  = 200,
) -> Tuple[int, float]:
    """
    Joue un épisode complet : agent (Noirs) vs opp (Blancs).
    Utilise OthelloEnv (interface Gymnasium) pour la boucle de jeu.

    À chaque tour :
      1. Agent sélectionne une action via epsilon-greedy
      2. env.step(a)  — coup de l'agent
      3. Adversaire sélectionne et joue (env.step)
      4. Transition (s, a, r, s', done) stockée dans le replay buffer
      5. Un pas de gradient (agent.update())

    Retourne : (winner ∈ {-1, 0, +1}, avg_loss)
    """
    env = OthelloEnv()
    env.reset()
    total_loss  = 0.0
    n_updates   = 0

    while not env.done:
        # ── Tour de l'agent (Noirs) ───────────────────────────────────────────
        board_before = env.state.board
        obs          = agent.board_to_obs(env.state.board, player=1)
        legal_ids    = env.legal_actions()           # [0..64], inclut PASS=64

        a = agent.select_action(obs, legal_ids)
        _, _, terminated, _, _ = env.step(a)

        if terminated:
            # Partie terminée après le coup de l'agent
            w        = get_winner(env.state.board)
            r        = float(w)                      # +1 / -1 / 0
            next_obs = agent.board_to_obs(env.state.board, player=1)
            agent.push(obs, a, r, next_obs, True, [64])
            loss = agent.update()
            if loss is not None:
                total_loss += loss; n_updates += 1
            break

        # ── Tour de l'adversaire (Blancs) ────────────────────────────────────
        mv_opp = opp.select_move(env.state.board, -1)
        _, _, terminated, _, _ = env.step(encode_action(mv_opp))

        # État suivant vu par l'agent (perspective Noirs)
        next_obs    = agent.board_to_obs(env.state.board, player=1)
        next_legal  = env.legal_actions()            # coups légaux pour Noirs

        if terminated:
            w = get_winner(env.state.board)
            r = float(w)
            agent.push(obs, a, r, next_obs, True, next_legal)
        else:
            r = shaped_reward(board_before, env.state.board, player=1) \
                if shaped else 0.0
            agent.push(obs, a, r, next_obs, False, next_legal)

        loss = agent.update()
        if loss is not None:
            total_loss += loss; n_updates += 1

    agent.decay_epsilon()
    return get_winner(env.state.board), total_loss / max(n_updates, 1)


# ── Évaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    agent:   DQNAgent,
    opp,
    n_games: int = 100,
) -> float:
    """
    Évalue l'agent (eps=0, Noirs) contre opp (Blancs).
    Retourne le taux de victoire des Noirs ∈ [0, 1].
    """
    saved_eps = agent.eps
    agent.eps = 0.0
    wins = 0

    for _ in range(n_games):
        env = OthelloEnv()
        env.reset()
        while not env.done:
            if env.state.player == 1:
                obs      = agent.board_to_obs(env.state.board, 1)
                legal_ids = env.legal_actions()
                a        = agent.select_action(obs, legal_ids)
                env.step(a)
            else:
                mv = opp.select_move(env.state.board, -1)
                env.step(encode_action(mv))

        if get_winner(env.state.board) == 1:
            wins += 1

    agent.eps = saved_eps
    return wins / n_games


# ── Main ────────────────────────────────────────────────────────────────────────

def main(n_episodes: int = 5_000) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("  DQN — CNN 3 canaux | Replay Buffer | Target Network")
    print(f"  Device : {device.upper()}")
    print(f"  {n_episodes} épisodes | curriculum 2 phases")
    print("=" * 65)

    agent = DQNAgent(
        lr=1e-4,
        gamma=0.99,
        eps=1.0,
        eps_min=0.05,
        eps_decay=0.9992,
        buffer_capacity=100_000,
        batch_size=256,
        target_update_freq=200,
        min_buffer_size=1_000,
        device=device,
        seed=0,
    )

    PHASE2_START = 1_001   # Random → MCTS
    PHASE3_START = 2_001   # MCTS → AlphaBeta-d2
    PHASE4_START = 3_001   # AlphaBeta-d2 → AlphaBeta-d3
    PHASE5_START = 4_001   # AlphaBeta-d3 → AlphaBeta-d4

    opp_random = _RandomAgent(seed=1)
    opp_mcts   = MCTSAgent(n_simulations=200, seed=1)
    opp_ab2    = AlphaBetaAgent(depth=2)
    opp_ab3    = AlphaBetaAgent(depth=3)
    opp_ab4    = AlphaBetaAgent(depth=4)

    log_every   = 200
    eval_every  = 500

    wins_total  = {1: 0, -1: 0, 0: 0}
    recent: List[int]   = []
    stats:  List[dict]  = []

    print(f"  Phase 1 (ep     1–{PHASE2_START - 1:4d}) : vs Random")
    print(f"  Phase 2 (ep {PHASE2_START:5d}–{PHASE3_START - 1:4d}) : vs MCTS-200")
    print(f"  Phase 3 (ep {PHASE3_START:5d}–{PHASE4_START - 1:4d}) : vs AlphaBeta-d2")
    print(f"  Phase 4 (ep {PHASE4_START:5d}–{PHASE5_START - 1:4d}) : vs AlphaBeta-d3")
    print(f"  Phase 5 (ep {PHASE5_START:5d}–{n_episodes:5d}) : vs AlphaBeta-d4")
    print("=" * 65)

    t0 = time.time()

    for ep in range(1, n_episodes + 1):
        if ep < PHASE2_START:
            opp, phase = opp_random, "Random"
        elif ep < PHASE3_START:
            opp, phase = opp_mcts, "MCTS"
        elif ep < PHASE4_START:
            opp, phase = opp_ab2, "AlphaBeta-d2"
        elif ep < PHASE5_START:
            opp, phase = opp_ab3, "AlphaBeta-d3"
        else:
            opp, phase = opp_ab4, "AlphaBeta-d4"

        w, avg_loss = play_episode(agent, opp, shaped=True)

        wins_total[w] += 1
        recent.append(w)
        if len(recent) > log_every:
            recent.pop(0)

        if ep % log_every == 0:
            wr = recent.count(1) / len(recent)
            elapsed = time.time() - t0
            em, es = divmod(int(elapsed), 60)
            print(
                f"Ep {ep:5d} [{phase:8s}] | ε={agent.eps:.4f}"
                f" | W={wins_total[1]} L={wins_total[-1]} D={wins_total[0]}"
                f" | WR last{log_every}={wr:.1%}"
                f" | loss={avg_loss:.4f}"
                f" | buf={len(agent.buffer)}"
                f" | +{em}m{es:02d}s"
            )
            stats.append({
                "ep":       ep,
                "phase":    phase,
                "win_rate": round(wr, 4),
                "eps":      round(agent.eps, 4),
                "avg_loss": round(avg_loss, 6),
            })

        if ep == PHASE2_START - 1:
            print()
            print("  *** Phase 2 : adversaire → MCTS-200 ***")
            print()
        elif ep == PHASE3_START - 1:
            print()
            print("  *** Phase 3 : adversaire → AlphaBeta-d2 ***")
            print()
        elif ep == PHASE4_START - 1:
            print()
            print("  *** Phase 4 : adversaire → AlphaBeta-d3 ***")
            print()
        elif ep == PHASE5_START - 1:
            print()
            print("  *** Phase 5 : adversaire → AlphaBeta-d4 ***")
            print()

        # Évaluation périodique
        if ep % eval_every == 0:
            wr_rand = evaluate(agent, _RandomAgent(seed=99), n_games=100)
            print(f"         [éval ep {ep}] vs Random={wr_rand:.1%}")

    # ── Sauvegarde ──────────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/dqn_model.pt"
    agent.save(model_path)
    print()
    print(f"Modèle sauvegardé → {model_path}")
    print(f"  Steps optimizer : {agent.steps}")
    print(f"  Buffer rempli   : {len(agent.buffer)}/{agent.buffer.capacity}")

    stats_path = "artifacts/dqn_training_stats.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ep", "phase", "win_rate", "eps", "avg_loss"]
        )
        writer.writeheader()
        writer.writerows(stats)
    print(f"  Stats CSV       → {stats_path}")

    # ── Évaluation finale ───────────────────────────────────────────────────
    agent.eps = 0.0

    print()
    print("Évaluation finale (eps=0) :")
    wr_rand = evaluate(agent, _RandomAgent(seed=99),               n_games=200)
    wr_mcts = evaluate(agent, MCTSAgent(n_simulations=200, seed=99), n_games=100)
    wr_ab2  = evaluate(agent, AlphaBetaAgent(depth=2),             n_games=100)
    wr_ab3  = evaluate(agent, AlphaBetaAgent(depth=3),             n_games=100)
    wr_ab4  = evaluate(agent, AlphaBetaAgent(depth=4),             n_games=100)
    print(f"  vs Random (200p)      : {wr_rand:.1%}")
    print(f"  vs MCTS-200 (100p)    : {wr_mcts:.1%}")
    print(f"  vs AlphaBeta-d2 (100p): {wr_ab2:.1%}")
    print(f"  vs AlphaBeta-d3 (100p): {wr_ab3:.1%}")
    print(f"  vs AlphaBeta-d4 (100p): {wr_ab4:.1%}")

    # ── Graphiques ──────────────────────────────────────────────────────────
    try:
        from src.experiments.plot_results import plot_dqn_learning_curve, plot_final_eval
        plot_dqn_learning_curve(csv_path=stats_path)
        plot_final_eval(
            {
                "vs Random (200p)":       wr_rand,
                "vs MCTS-200 (100p)":     wr_mcts,
                "vs AlphaBeta-d2 (100p)": wr_ab2,
                "vs AlphaBeta-d3 (100p)": wr_ab3,
                "vs AlphaBeta-d4 (100p)": wr_ab4,
            },
            out_dir="artifacts",
            prefix="dqn_",
        )
    except Exception as exc:
        print(f"  [plot] {exc}")

    total = time.time() - t0
    th, rem = divmod(int(total), 3600)
    tm, ts = divmod(rem, 60)
    dur_str = (f"{th}h {tm}m {ts:02d}s" if th else f"{tm}m {ts:02d}s")
    print()
    print(f"Durée totale entraînement DQN : {dur_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5_000,
                        help="Nombre d'épisodes (défaut : 5 000)")
    args = parser.parse_args()
    main(n_episodes=args.episodes)
