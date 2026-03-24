# src/rl_v2/features_v2.py
"""
Représentation d'état enrichie pour le Q-Learning v2.

Améliorations vs features.py v1 :
  - Ajout d'un score positionnel (poids WEIGHTS_FLAT, 64 cases) comme 5e feature
  - Fonction shaped_reward() pour les récompenses intermédiaires (coins + mobilité)

État = 5-tuple (diff_pions, diff_mobilité, diff_coins, phase, score_positionnel)
Espace d'états max ≈ 33 × 21 × 9 × 3 × 25 = 469 350 (très gérable en Q-table).
"""
from __future__ import annotations
from typing import Tuple, Optional, List

from src.envs.othello_env import get_legal_moves

Move = Tuple[int, int]
Board = Tuple[int, int]  # (black_bb, white_bb)

# ── Re-exports de v1 (pour ne pas casser train_rl_v2 qui les importe ici) ────
from src.rl.features import action_to_id, id_to_action  # noqa: F401

# ── Constantes ────────────────────────────────────────────────────────────────

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# Poids positionnels standard Othello (row-major, ligne 0 = haut du plateau).
# Coins très précieux (+120), cases X "suicide" très pénalisées (-40), etc.
WEIGHTS_FLAT = [
    120, -20,  20,   5,   5,  20, -20, 120,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
     20,  -5,  15,   3,   3,  15,  -5,  20,
      5,  -5,   3,   3,   3,   3,  -5,   5,
      5,  -5,   3,   3,   3,   3,  -5,   5,
     20,  -5,  15,   3,   3,  15,  -5,  20,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
    120, -20,  20,   5,   5,  20, -20, 120,
]

# ── Type de l'état ─────────────────────────────────────────────────────────────

State = Tuple[int, int, int, int, int]


# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def _bit_at(bb: int, r: int, c: int) -> int:
    return (bb >> (r * 8 + c)) & 1


def _count_bits(bb: int) -> int:
    return int(bb).bit_count()


def _disc(x: int, step: int, lo: int, hi: int) -> int:
    """Discrétise x par pas step et clip dans [lo, hi]."""
    v = x // step
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _corners_diff(board: Board, player: int) -> int:
    """Différence de coins occupés (my - opp), ∈ [-4, 4]."""
    black_bb, white_bb = board
    my = opp = 0
    for r, c in CORNERS:
        if player == 1:
            if _bit_at(black_bb, r, c): my += 1
            if _bit_at(white_bb, r, c): opp += 1
        else:
            if _bit_at(white_bb, r, c): my += 1
            if _bit_at(black_bb, r, c): opp += 1
    return my - opp


def _positional_score(board: Board, player: int) -> int:
    """
    Score positionnel pondéré : somme(WEIGHTS * mes pions) - somme(WEIGHTS * pions adverses).
    Plage pratique : environ [-500, +500].
    """
    black_bb, white_bb = board
    score = 0
    for i in range(64):
        w = WEIGHTS_FLAT[i]
        if (black_bb >> i) & 1:
            score += w if player == 1 else -w
        elif (white_bb >> i) & 1:
            score -= w if player == 1 else -w
    return score


# ── Feature principale ─────────────────────────────────────────────────────────

def state_features_v2(board: Board, player: int) -> State:
    """
    Retourne un état discret (5-tuple) — clé de la Q-table v2.

    Features (toutes du point de vue de `player`) :
      1. diff_pions      my - opp, pas=4, clip [-16, 16]  → 33 valeurs
      2. diff_mobilité   my_moves - opp_moves, pas=2, clip [-10, 10] → 21 valeurs
      3. diff_coins      my - opp ∈ [-4, 4]              →  9 valeurs
      4. phase           0=début(<20), 1=milieu, 2=fin(≥50) → 3 valeurs
      5. score_pos       pondéré, pas=50, clip [-12, 12]  → 25 valeurs
    """
    black_bb, white_bb = board
    nb = _count_bits(black_bb)
    nw = _count_bits(white_bb)

    my_p  = nb if player == 1 else nw
    opp_p = nw if player == 1 else nb
    diff_p = my_p - opp_p

    my_moves  = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, -player))
    diff_m = my_moves - opp_moves

    diff_c = _corners_diff(board, player)

    filled = nb + nw
    phase = 0 if filled < 20 else (1 if filled < 50 else 2)

    pos = _positional_score(board, player)

    return (
        _disc(diff_p, step=4,  lo=-16, hi=16),
        _disc(diff_m, step=2,  lo=-10, hi=10),
        diff_c,
        phase,
        _disc(pos,   step=50, lo=-12, hi=12),
    )


# ── Récompense intermédiaire (reward shaping) ─────────────────────────────────

def shaped_reward(board_before: Board, board_after: Board, player: int) -> float:
    """
    Récompense intermédiaire (pas de fin de partie) basée sur la variation de
    situation entre board_before (avant l'action de `player`) et board_after
    (après la réponse de l'adversaire).

    Composantes :
      - Bonus coin  : ±0.3 par coin gagné/perdu  (fort signal, rare)
      - Bonus mob   : différence de mobilité × 0.01  (signal léger et fréquent)

    Ces valeurs sont petites par rapport au reward terminal (±1.0).
    """
    corner_delta = _corners_diff(board_after, player) - _corners_diff(board_before, player)
    corner_bonus = corner_delta * 0.3

    my_moves  = len(get_legal_moves(board_after, player))
    opp_moves = len(get_legal_moves(board_after, -player))
    mob_bonus = (my_moves - opp_moves) * 0.01

    return corner_bonus + mob_bonus
