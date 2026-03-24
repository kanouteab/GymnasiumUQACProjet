# src/experiments/train_rl.py
"""
Entraînement Q-Learning v2 avec toutes les améliorations.

Améliorations vs train_rl.py v1 :
  1. État enrichi (score positionnel)  → via state_features_v2()
  2. Double Q-Learning                 → via QLearningAgentV2
  3. Reward shaping (coins + mobilité) → récompenses intermédiaires non nulles
  4. Self-play (~30 % des épisodes)    → l'agent apprend les deux couleurs
  5. Curriculum learning               → ep 1-2000 vs Random, ep 2001-5000 vs Alpha-Beta depth=2
  6. 5 000 épisodes (vs 2 000 en v1)  → plus de données d'entraînement
  7. Évaluation périodique vs Random ET vs Alpha-Beta

Lancement :
    python -m src.experiments.train_rl
"""
from __future__ import annotations

import pickle
import os
import random
from typing import Optional, Tuple, List

from src.envs.othello_env import (
    initial_board, get_legal_moves, apply_move,
    is_terminal, get_winner,
)
from Atelier2.GymnasiumUQACProjet.src.rl.features import (
    state_features_v2, shaped_reward, action_to_id, id_to_action,
)
from Atelier2.GymnasiumUQACProjet.src.rl.qlearning import QLearningAgentV2
from src.agents.alphabeta import AlphaBetaAgent

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


def legal_action_ids(board: Board, player: int) -> List[int]:
    legal = get_legal_moves(board, player)
    if not legal:
        return [-1]   # PASS
    return [action_to_id(mv) for mv in legal]


# Mode 1 : agent (Noir) vs adversaire quelconque (Blanc)
# Fonctionne avec RandomAgent, AlphaBetaAgent, ou tout objet ayant
# select_move(board, player) -> Optional[Move].

def play_episode_vs_opp(
    agent: QLearningAgentV2,
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
    """
    board = initial_board()
    steps = 0

    while not is_terminal(board) and steps < max_steps:

        # Tour de l'agent (Noir) 
        board_before = board
        s = state_features_v2(board, player=1)
        legal_a = legal_action_ids(board, player=1)
        a = agent.select_action(s, legal_a)
        mv = id_to_action(a)
        if mv is not None:
            board = apply_move(board, 1, mv)
        steps += 1

        done = is_terminal(board)

        # Tour de l'adversaire (Blanc), si la partie continue 
        if not done and steps < max_steps:
            mv_opp = opp.select_move(board, -1)
            if mv_opp is not None:
                board = apply_move(board, -1, mv_opp)
            steps += 1
            done = is_terminal(board)

        # Mise à jour TD
        s2      = state_features_v2(board, player=1)
        legal_a2 = legal_action_ids(board, player=1)

        if done:
            w = get_winner(board)
            r = 1.0 if w == 1 else (-1.0 if w == -1 else 0.0)
            agent.update(s, a, r, s2, legal_a2, done=True)
        else:
            r = shaped_reward(board_before, board, player=1) if shaped else 0.0
            agent.update(s, a, r, s2, legal_a2, done=False)

    agent.decay_epsilon()
    return get_winner(board)


# Alias rétrocompatible
def play_episode_vs_random(agent, opp, seed, shaped=True, max_steps=200):
    return play_episode_vs_opp(agent, opp, seed, shaped, max_steps)


# Mode 2 : self-play (agent joue les deux couleurs)

def play_episode_selfplay(
    agent: QLearningAgentV2,
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
    """
    board   = initial_board()
    player  = 1
    steps   = 0

    # Pour chaque joueur : (s, a, board_avant_son_coup) en attente d'update
    deferred: dict = {1: None, -1: None}

    while not is_terminal(board) and steps < max_steps:

        board_before = board
        s       = state_features_v2(board, player)
        legal_a = legal_action_ids(board, player)
        a       = agent.select_action(s, legal_a)
        mv      = id_to_action(a)
        if mv is not None:
            board = apply_move(board, player, mv)
        steps  += 1

        done  = is_terminal(board)
        other = -player

        # Mettre à jour la transition en attente de l'adversaire
        # (maintenant qu'il vient d'agir, on connaît s' pour lui)
        if deferred[other] is not None:
            s_o, a_o, bb_o = deferred[other]
            s2_o   = state_features_v2(board, other)
            la2_o  = legal_action_ids(board, other)

            if done:
                w   = get_winner(board)
                r_o = 1.0 if w == other else (-1.0 if w == -other else 0.0)
            elif shaped:
                r_o = shaped_reward(bb_o, board, other)
            else:
                r_o = 0.0

            agent.update(s_o, a_o, r_o, s2_o, la2_o, done=done)
            if done:
                deferred[other] = None

        # Mettre à jour le joueur courant si partie terminée
        if done:
            w = get_winner(board)
            r = 1.0 if w == player else (-1.0 if w == -player else 0.0)
            s2  = state_features_v2(board, player)
            la2 = legal_action_ids(board, player)
            agent.update(s, a, r, s2, la2, done=True)
        else:
            # Stocker pour mise à jour différée
            deferred[player] = (s, a, board_before)

        player = other

    agent.decay_epsilon()
    return get_winner(board)


# Évaluation

def evaluate(agent: QLearningAgentV2, opp=None, n_games: int = 200, seed: int = 99999) -> float:
    """
    Joue n_games parties : agent (Noirs, eps=0) vs opp (Blancs).
    opp par défaut = RandomAgent. Passe un AlphaBetaAgent pour évaluer contre lui.
    Retourne le taux de victoire des Noirs (∈ [0, 1]).
    """
    saved_eps = agent.eps
    agent.eps = 0.0  # mode exploitation pur
    if opp is None:
        opp = RandomAgent(seed=seed)
    wins = 0

    for i in range(n_games):
        board = initial_board()
        steps = 0
        while not is_terminal(board) and steps < 200:
            # Agent (Noir)
            s = state_features_v2(board, 1)
            legal_a = legal_action_ids(board, 1)
            a = agent.select_action(s, legal_a)
            mv = id_to_action(a)
            if mv is not None:
                board = apply_move(board, 1, mv)
            steps += 1
            if is_terminal(board):
                break
            # Adversaire (Blanc)
            mv_opp = opp.select_move(board, -1)
            if mv_opp is not None:
                board = apply_move(board, -1, mv_opp)
            steps += 1

        if get_winner(board) == 1:
            wins += 1

    agent.eps = saved_eps
    return wins / n_games


# Main

def main() -> None:
    agent = QLearningAgentV2(
        alpha=0.2,
        gamma=0.95,
        eps=1.0,
        eps_decay=0.9993,   # atteint eps_min ≈ 0.05 après ~1 000 épisodes
        eps_min=0.05,
        seed=0,
    )
    """
    Phases d'entraînement :
    Phase 1 (ep 1-2000)    : Random             → l'agent apprend les bases
    Phase 2 (ep 2001-5000) : MCTS               → adversaire stratégique intermédiaire
    Phase 3 (ep 5001-8000) : Alpha-Beta d2      → adversaire stratégique intermédiaire
    Phase 4 (ep 8001-10000) : Alpha-Beta d3     → adversaire fort, affinage fin
    Phase 5 (ep 10001-12000) : Alpha-Beta d4    → adversaire très fort, test ultime
    depth=2 : assez fort pour enseigner mais pas écrasant (~30-40% win rate attendu)
    depth=3 : fort, l'agent doit être déjà bien entraîné pour espérer rivaliser (~15-30% win rate attendu)
    depth=4 : adversaire sérieux, l'agent ne gagnera que ~10-20%, mais même ce rare signal positif est très informatif.
    """
    opp_random = RandomAgent(seed=1)
    opp_ab2    = AlphaBetaAgent(depth=2, use_move_ordering=True)
    opp_ab4    = AlphaBetaAgent(depth=4, use_move_ordering=True)
    CURRICULUM_SWITCH_1 = 2000   # Random → Alpha-Beta d2
    CURRICULUM_SWITCH_2 = 5000   # Alpha-Beta d2 → Alpha-Beta d4

    n_episodes   = 8000
    eval_every   = 500   # évaluation intermédiaire toutes les N épisodes
    log_every    = 200   # log compact toutes les N épisodes

    wins_total = {1: 0, -1: 0, 0: 0}
    recent: List[int] = []

    print("=" * 65)
    print("  Q-Learning v2 — Double Q + Reward Shaping + Curriculum + Self-Play")
    print(f"  {n_episodes} épisodes | alpha={agent.alpha} gamma={agent.gamma}")
    print(f"  Phase 1 (ep 1-{CURRICULUM_SWITCH_1})                : vs Random")
    print(f"  Phase 2 (ep {CURRICULUM_SWITCH_1+1}-{CURRICULUM_SWITCH_2})             : vs Alpha-Beta depth=2")
    print(f"  Phase 3 (ep {CURRICULUM_SWITCH_2+1}-{n_episodes})             : vs Alpha-Beta depth=4")
    print("=" * 65)

    for ep in range(1, n_episodes + 1):

        # Choix de l'adversaire selon la phase du curriculum
        if ep <= CURRICULUM_SWITCH_1:
            opp = opp_random
        elif ep <= CURRICULUM_SWITCH_2:
            opp = opp_ab2
        else:
            opp = opp_ab4

        # 30 % self-play, 70 % vs adversaire courant
        use_selfplay = (ep % 10) < 3

        if use_selfplay:
            w = play_episode_selfplay(agent, seed=ep, shaped=True)
        else:
            w = play_episode_vs_opp(agent, opp, seed=ep, shaped=True)

        wins_total[w] += 1
        recent.append(w)
        if len(recent) > log_every:
            recent.pop(0)

        if ep % log_every == 0:
            wr    = recent.count(1) / len(recent)
            if ep <= CURRICULUM_SWITCH_1:
                phase = "Random"
            elif ep <= CURRICULUM_SWITCH_2:
                phase = "AlphaBeta-d2"
            else:
                phase = "AlphaBeta-d4"
            print(
                f"Ep {ep:5d} [{phase:12s}] | ε={agent.eps:.4f} | "
                f"W={wins_total[1]} L={wins_total[-1]} D={wins_total[0]} | "
                f"WinRate last{log_every}={wr:.1%} | "
                f"Q1={len(agent.Q1)} Q2={len(agent.Q2)} entrées"
            )

        if ep % eval_every == 0:
            wr_rand = evaluate(agent, opp=None, n_games=200)
            wr_ab2  = evaluate(agent, opp=AlphaBetaAgent(depth=2), n_games=100)
            wr_ab4  = evaluate(agent, opp=AlphaBetaAgent(depth=4), n_games=100)
            print(
                f"  >> Éval ep {ep} : "
                f"{wr_rand:.1%} vs Random (200p) | "
                f"{wr_ab2:.1%} vs AB-d2 (100p) | "
                f"{wr_ab4:.1%} vs AB-d4 (100p)"
            )

        # Annonces des switchs de curriculum
        if ep == CURRICULUM_SWITCH_1:
            print()
            print(f"  *** Phase 2 : adversaire → Alpha-Beta depth=2 ***")
            print()
        if ep == CURRICULUM_SWITCH_2:
            print()
            print(f"  *** Phase 3 : adversaire → Alpha-Beta depth=4 ***")
            print()

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/qtable_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump({"Q1": agent.Q1, "Q2": agent.Q2}, f)

    print()
    print(f"Sauvegardé → {path}")
    print(f"  Q1 : {len(agent.Q1)} entrées")
    print(f"  Q2 : {len(agent.Q2)} entrées")
    wr_rand_final = evaluate(agent, opp=None,                    n_games=500)
    wr_ab2_final  = evaluate(agent, opp=AlphaBetaAgent(depth=2), n_games=200)
    wr_ab4_final  = evaluate(agent, opp=AlphaBetaAgent(depth=4), n_games=200)
    print(f"  Win rate final vs Random (500p)       : {wr_rand_final:.1%}")
    print(f"  Win rate final vs AlphaBeta-d2 (200p) : {wr_ab2_final:.1%}")
    print(f"  Win rate final vs AlphaBeta-d4 (200p) : {wr_ab4_final:.1%}")


if __name__ == "__main__":
    main()
