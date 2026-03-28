# src/experiments/train_rl.py
"""
Entraînement Q-Learning avec toutes les améliorations.

Améliorations vs v1 :
  1. État enrichi (score positionnel)  → via state_features()
  2. Double Q-Learning                 → via QLearningAgent
  3. Reward shaping (coins + mobilité) → récompenses intermédiaires non nulles
  4. Self-play (0% Phase 1, ~30% phases avancées)
  5. Curriculum adaptatif basé sur le win rate :
       Phase 1 : vs Random   (seuil 80%, min 2000 ep, max 8000 ep)
       Phase 2 : vs MCTS-50  (seuil 40%, min 1000 ep, max 5000 ep)
       Phase 3 : vs MCTS-200 (seuil 25%, min 1500 ep, max 5000 ep)
       Phase 4 : vs AB-d2    (seuil 20%, min 1000 ep, max 4000 ep)
       Phase 5 : vs AB-d3    (seuil 15%, min 1000 ep, max 4000 ep)
       Phase 6 : vs AB-d4    (finale,    min 1000 ep, max jusqu'au 20000 totales)
  6. Epsilon reseté à chaque changement de phase (injection de curiosité)
  7. Plafond global de 20 000 épisodes

Lancement :
    python -m src.experiments.train_rl
"""
from __future__ import annotations

import pickle
import csv
import os
import random
import time
from typing import Optional, Tuple, List

from src.envs.othello_env import (
    get_legal_moves, get_winner,
    encode_action, OthelloEnv,
)
from src.training.features import (
    state_features, shaped_reward,
)
from src.agents.qlearning import QLearningAgent
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.mcts import MCTSAgent

Move  = Tuple[int, int]
Board = Tuple[int, int]


# Utilitaires communs 

class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        return self.rng.choice(legal)


# Mode 1 : agent (Noir) vs adversaire quelconque (Blanc)
# Fonctionne avec RandomAgent, AlphaBetaAgent, ou tout objet ayant
# select_move(board, player) -> Optional[Move].

def play_episode_vs_opp(
    agent: QLearningAgent,
    opp,
    seed: int,
    shaped: bool = True,
    max_steps: int = 200,
) -> int:
    """
    L'agent joue toujours les Noirs (+1).
    Chaque "pas" = 1 coup agent + 1 coup adversaire (groupés en un step TD).
    Récompense intermédiaire via shaped_reward() si shaped=True.
    Fonctionne avec n'importe quel adversaire implémentant select_move().
    Utilise OthelloEnv (interface Gymnasium) pour la boucle de jeu.
    """
    env = OthelloEnv()
    env.reset()
    steps = 0

    while not env.done and steps < max_steps:

        # Tour de l'agent (Noir)
        board_before = env.state.board
        s = state_features(env.state.board, player=1)
        legal_a = env.legal_actions()
        a = agent.select_action(s, legal_a)
        _, _, terminated, _, _ = env.step(a)
        steps += 1

        # Tour de l'adversaire (Blanc), si la partie continue
        if not terminated and steps < max_steps:
            mv_opp = opp.select_move(env.state.board, -1)
            _, _, terminated, _, _ = env.step(encode_action(mv_opp))
            steps += 1

        # Mise à jour TD
        s2       = state_features(env.state.board, player=1)
        legal_a2 = env.legal_actions()

        if terminated:
            w = get_winner(env.state.board)
            r = 1.0 if w == 1 else (-1.0 if w == -1 else 0.0)
            agent.update(s, a, r, s2, legal_a2, done=True)
        else:
            r = shaped_reward(board_before, env.state.board, player=1) if shaped else 0.0
            agent.update(s, a, r, s2, legal_a2, done=False)

    agent.decay_epsilon()
    return get_winner(env.state.board)


# Alias rétrocompatible
def play_episode_vs_random(agent, opp, seed, shaped=True, max_steps=200):
    return play_episode_vs_opp(agent, opp, seed, shaped, max_steps)


# Mode 2 : self-play (agent joue les deux couleurs)

def play_episode_selfplay(
    agent: QLearningAgent,
    seed: int,
    shaped: bool = True,
    max_steps: int = 200,
) -> int:
    """
    Le même agent (même Q1/Q2) joue les deux couleurs.
    Les features étant relatives au joueur courant, une seule Q-table
    suffit pour généraliser la stratégie aux deux camps.

    Algorithme : mise à jour TD différée (deferred update).
      - Quand le joueur P agit, on ne connaît pas encore s' pour P
        (l'adversaire n'a pas encore répondu).
      - On stocke (s, a) pour P et on met à jour P lors du prochain
        tour de P (quand l'adversaire vient d'agir → s' est connu).
    Utilise OthelloEnv (interface Gymnasium) pour la boucle de jeu.
    """
    env = OthelloEnv()
    env.reset()
    steps = 0

    # Pour chaque joueur : (s, a, board_avant_son_coup) en attente d'update
    deferred: dict = {1: None, -1: None}

    while not env.done and steps < max_steps:
        player = env.state.player
        other  = -player
        board_before = env.state.board

        s       = state_features(env.state.board, player)
        legal_a = env.legal_actions()
        a       = agent.select_action(s, legal_a)

        _, _, terminated, _, _ = env.step(a)
        steps += 1

        # Après env.step(a), env.state.player == other (joueur switché)
        # env.legal_actions() retourne donc les coups légaux pour `other`

        # Mettre à jour la transition en attente de l'adversaire
        # (maintenant qu'il vient d'agir, on connaît s' pour lui)
        if deferred[other] is not None:
            s_o, a_o, bb_o = deferred[other]
            s2_o  = state_features(env.state.board, other)
            la2_o = env.legal_actions()  # env.state.player == other ici

            if terminated:
                w   = get_winner(env.state.board)
                r_o = 1.0 if w == other else (-1.0 if w == -other else 0.0)
            elif shaped:
                r_o = shaped_reward(bb_o, env.state.board, other)
            else:
                r_o = 0.0

            agent.update(s_o, a_o, r_o, s2_o, la2_o, done=terminated)
            if terminated:
                deferred[other] = None

        # Mettre à jour le joueur courant si partie terminée
        if terminated:
            w = get_winner(env.state.board)
            r = 1.0 if w == player else (-1.0 if w == -player else 0.0)
            s2  = state_features(env.state.board, player)
            la2 = env.legal_actions()
            agent.update(s, a, r, s2, la2, done=True)
        else:
            # Stocker pour mise à jour différée
            deferred[player] = (s, a, board_before)

    agent.decay_epsilon()
    return get_winner(env.state.board)


# Évaluation

def evaluate(agent: QLearningAgent, opp=None, n_games: int = 200, seed: int = 99999) -> float:
    """
    Joue n_games parties : agent (Noirs, eps=0) vs opp (Blancs).
    opp par défaut = RandomAgent. Passe un AlphaBetaAgent pour évaluer contre lui.
    Retourne le taux de victoire des Noirs (∈ [0, 1]).
    Utilise OthelloEnv (interface Gymnasium) pour la boucle de jeu.
    """
    saved_eps = agent.eps
    agent.eps = 0.0  # mode exploitation pur
    if opp is None:
        opp = RandomAgent(seed=seed)
    wins = 0

    for i in range(n_games):
        env = OthelloEnv()
        env.reset()
        steps = 0
        while not env.done and steps < 200:
            if env.state.player == 1:
                # Agent (Noir)
                s = state_features(env.state.board, 1)
                legal_a = env.legal_actions()
                a = agent.select_action(s, legal_a)
                env.step(a)
            else:
                # Adversaire (Blanc)
                mv_opp = opp.select_move(env.state.board, -1)
                env.step(encode_action(mv_opp))
            steps += 1

        if get_winner(env.state.board) == 1:
            wins += 1

    agent.eps = saved_eps
    return wins / n_games


# Main

def main() -> None:
    agent = QLearningAgent(
        alpha=0.2,
        gamma=0.95,
        eps=1.0,
        eps_decay=0.9993,
        eps_min=0.05,
        seed=0,
    )

    opp_random  = RandomAgent(seed=1)
    opp_mcts50  = MCTSAgent(n_simulations=50,  seed=1)
    opp_mcts200 = MCTSAgent(n_simulations=200, seed=1)
    opp_ab2     = AlphaBetaAgent(depth=2, use_move_ordering=True)
    opp_ab3     = AlphaBetaAgent(depth=3, use_move_ordering=True)
    opp_ab4     = AlphaBetaAgent(depth=4, use_move_ordering=True)

    # Curriculum adaptatif : sortie par win-rate ou plafond de sécurité.
    #   selfplay  : fraction d'épisodes joués en self-play (0 = aucun)
    #   eps_reset : eps minimum forcé à l'entrée de la phase (None = premier départ)
    PHASES = [
        {"name": "Random",       "opp": opp_random,  "win_threshold": 0.80, "min_eps": 2000, "max_eps": 8000, "selfplay": 0.0,  "eps_reset": None},
        {"name": "MCTS-50",      "opp": opp_mcts50,  "win_threshold": 0.45, "min_eps": 1000, "max_eps": 5000, "selfplay": 0.30, "eps_reset": 0.60},
        {"name": "AlphaBeta-d2", "opp": opp_ab2,     "win_threshold": 0.35, "min_eps": 1000, "max_eps": 5000, "selfplay": 0.30, "eps_reset": 0.50},
        {"name": "AlphaBeta-d3", "opp": opp_ab3,     "win_threshold": 0.25, "min_eps": 1000, "max_eps": 4000, "selfplay": 0.30, "eps_reset": 0.40},
        {"name": "MCTS-200",     "opp": opp_mcts200, "win_threshold": 0.20, "min_eps": 1500, "max_eps": 4000, "selfplay": 0.30, "eps_reset": 0.30},
        {"name": "AlphaBeta-d4", "opp": opp_ab4,     "win_threshold": None, "min_eps": 1000, "max_eps": None, "selfplay": 0.30, "eps_reset": 0.20},
    ]

    TOTAL_EPISODES = 50_000
    log_every      = 200   # log compact toutes les N épisodes

    wins_total = {1: 0, -1: 0, 0: 0}
    recent:        List[int] = []   # tous épisodes (affichage)
    recent_vs_opp: List[int] = []   # hors self-play (critère de sortie, signal non pollué)
    stats:         List[dict] = []

    current_phase = 0
    phase_ep      = 0   # épisodes joués dans la phase courante

    print("=" * 65)
    print("  Q-Learning — Double Q + Reward Shaping + Curriculum adaptatif + Self-Play")
    print(f"  Max {TOTAL_EPISODES} épisodes | alpha={agent.alpha} gamma={agent.gamma}")
    for i, ph in enumerate(PHASES):
        thr = f"{ph['win_threshold']:.0%}" if ph['win_threshold'] is not None else "—"
        sp  = f"{int(ph['selfplay'] * 100)}%"
        mx  = str(ph['max_eps']) if ph['max_eps'] is not None else "total"
        print(f"  Phase {i+1}: {ph['name']:12s} | seuil={thr:4s} | min={ph['min_eps']:4d} | max={mx:>5} | selfplay={sp}")
    print("=" * 65)

    t0 = time.time()

    for ep_total in range(1, TOTAL_EPISODES + 1):
        ph      = PHASES[current_phase]
        opp     = ph["opp"]
        sp_frac = ph["selfplay"]

        # 0% self-play en phase 1, 30% en phases suivantes
        use_selfplay = (sp_frac > 0.0) and ((ep_total % 10) < int(sp_frac * 10))

        if use_selfplay:
            w = play_episode_selfplay(agent, seed=ep_total, shaped=True)
        else:
            w = play_episode_vs_opp(agent, opp, seed=ep_total, shaped=True)

        wins_total[w] += 1
        recent.append(w)
        if len(recent) > log_every:
            recent.pop(0)

        if not use_selfplay:
            recent_vs_opp.append(w)
            if len(recent_vs_opp) > log_every:
                recent_vs_opp.pop(0)

        phase_ep += 1

        # ── Critère de sortie de phase ──────────────────────────────────────
        if current_phase < len(PHASES) - 1:
            wr_opp  = recent_vs_opp.count(1) / len(recent_vs_opp) if recent_vs_opp else 0.0
            win_thr = ph["win_threshold"]
            max_ep  = ph["max_eps"]
            min_ep  = ph["min_eps"]

            crit_wr = (
                phase_ep >= min_ep
                and len(recent_vs_opp) >= log_every
                and win_thr is not None
                and wr_opp >= win_thr
            )
            crit_cap = (max_ep is not None and phase_ep >= max_ep)

            if crit_wr or crit_cap:
                reason = "win-rate" if crit_wr else "plafond"
                current_phase += 1
                phase_ep       = 0
                recent_vs_opp  = []
                next_ph        = PHASES[current_phase]
                if next_ph["eps_reset"] is not None:
                    agent.eps = max(agent.eps, next_ph["eps_reset"])
                print()
                print(
                    f"  *** Phase {current_phase + 1} : adversaire → {next_ph['name']} "
                    f"(raison : {reason}, ε→{agent.eps:.2f}) ***"
                )
                print()

        # ── Log périodique ───────────────────────────────────────────────────
        if ep_total % log_every == 0:
            wr     = recent.count(1) / len(recent)
            wr_opp = recent_vs_opp.count(1) / len(recent_vs_opp) if recent_vs_opp else 0.0
            phase_name = PHASES[current_phase]["name"]
            elapsed = time.time() - t0
            em, es  = divmod(int(elapsed), 60)
            print(
                f"Ep {ep_total:5d} [{phase_name:12s}] | ε={agent.eps:.4f} | "
                f"W={wins_total[1]} L={wins_total[-1]} D={wins_total[0]} | "
                f"WR(all)={wr:.1%} WR(vs-opp)={wr_opp:.1%} | "
                f"États={agent.n_visited()} | "
                f"+{em}m{es:02d}s"
            )
            stats.append({
                "ep":        ep_total,
                "phase":     phase_name,
                "win_rate":  round(wr, 4),
                "wr_vs_opp": round(wr_opp, 4),
                "eps":       round(agent.eps, 4),
            })

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/qtable.pkl"
    with open(path, "wb") as f:
        pickle.dump({"Q1": agent.Q1, "Q2": agent.Q2}, f)

    print()
    print(f"Sauvegardé → {path}")
    print(f"  États visités : {agent.n_visited()}")

    stats_path = "artifacts/training_stats.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ep", "phase", "win_rate", "wr_vs_opp", "eps"])
        writer.writeheader()
        writer.writerows(stats)
    print(f"  Stats d'entraînement → {stats_path}")

    try:
        from src.experiments.plot_results import plot_learning_curve
        plot_learning_curve(csv_path=stats_path)
    except Exception as exc:
        print(f"  [plot] Impossible de générer la courbe : {exc}")

    wr_rand_final = evaluate(agent, opp=None,                                  n_games=500)
    wr_mcts_final = evaluate(agent, opp=MCTSAgent(n_simulations=200, seed=1), n_games=200)
    wr_ab2_final  = evaluate(agent, opp=AlphaBetaAgent(depth=2),              n_games=200)
    wr_ab3_final  = evaluate(agent, opp=AlphaBetaAgent(depth=3),              n_games=200)
    wr_ab4_final  = evaluate(agent, opp=AlphaBetaAgent(depth=4),              n_games=200)
    print(f"  Win rate final vs Random (500p)       : {wr_rand_final:.1%}")
    print(f"  Win rate final vs MCTS-200 (200p)     : {wr_mcts_final:.1%}")
    print(f"  Win rate final vs AlphaBeta-d2 (200p) : {wr_ab2_final:.1%}")
    print(f"  Win rate final vs AlphaBeta-d3 (200p) : {wr_ab3_final:.1%}")
    print(f"  Win rate final vs AlphaBeta-d4 (200p) : {wr_ab4_final:.1%}")

    try:
        from src.experiments.plot_results import plot_final_eval
        plot_final_eval({
            "vs Random (500p)":       wr_rand_final,
            "vs MCTS-200 (200p)":     wr_mcts_final,
            "vs AlphaBeta-d2 (200p)": wr_ab2_final,
            "vs AlphaBeta-d3 (200p)": wr_ab3_final,
            "vs AlphaBeta-d4 (200p)": wr_ab4_final,
        })
    except Exception as exc:
        print(f"  [plot] Impossible de générer le graphique final : {exc}")

    total = time.time() - t0
    th, rem = divmod(int(total), 3600)
    tm, ts = divmod(rem, 60)
    dur_str = (f"{th}h {tm}m {ts:02d}s" if th else f"{tm}m {ts:02d}s")
    print()
    print(f"Durée totale entraînement QL : {dur_str}")


if __name__ == "__main__":
    main()
