# src/rl/features.py
from __future__ import annotations
from typing import Tuple, Optional, List

from src.envs.othello_env import get_legal_moves

Move = Tuple[int, int]
Board = Tuple[int, int]  # (black_bb, white_bb)

CORNERS = [(0,0),(0,7),(7,0),(7,7)]

def _bit_at(bb: int, r: int, c: int) -> int:
    return (bb >> (r * 8 + c)) & 1

def _count_bits(bb: int) -> int:
    return int(bb).bit_count()

def _corners_diff(board: Board, player: int) -> int:
    black_bb, white_bb = board
    my = 0
    opp = 0
    for r,c in CORNERS:
        if player == 1:
            if _bit_at(black_bb, r, c): my += 1
            if _bit_at(white_bb, r, c): opp += 1
        else:
            if _bit_at(white_bb, r, c): my += 1
            if _bit_at(black_bb, r, c): opp += 1
    return my - opp

def _disc(x: int, step: int, lo: int, hi: int) -> int:
    """Discrétise x par pas 'step' et clip dans [lo,hi]."""
    v = x // step
    if v < lo: return lo
    if v > hi: return hi
    return v

def state_features(board: Board, player: int) -> Tuple[int, int, int, int]:
    """
    Retourne un état discret (petit tuple) utilisable comme clé de Q-table.
    Features:
      - diff pions (my - opp) discretisé
      - diff mobilité (my_moves - opp_moves) discretisé
      - diff coins (my - opp) ∈ [-4..4]
      - phase (0 début / 1 milieu / 2 fin)
    """
    black_bb, white_bb = board
    nb = _count_bits(black_bb)
    nw = _count_bits(white_bb)

    my_p = nb if player == 1 else nw
    opp_p = nw if player == 1 else nb
    diff_p = my_p - opp_p  # [-64..64]

    my_moves = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, -player))
    diff_m = my_moves - opp_moves  # ~[-20..20]

    diff_c = _corners_diff(board, player)  # [-4..4]

    filled = nb + nw
    if filled < 20:
        phase = 0
    elif filled < 50:
        phase = 1
    else:
        phase = 2

    # discrétisation (simple et robuste)
    diff_p_d = _disc(diff_p, step=4, lo=-16, hi=16)
    diff_m_d = _disc(diff_m, step=2, lo=-10, hi=10)
    diff_c_d = diff_c  # déjà petit
    return (diff_p_d, diff_m_d, diff_c_d, phase)

def action_to_id(move: Optional[Move]) -> int:
    """Encode move -> int. PASS(None) -> -1, sinon r*8+c."""
    if move is None:
        return -1
    r, c = move
    return r * 8 + c

def id_to_action(aid: int) -> Optional[Move]:
    """Decode int -> move."""
    if aid == -1:
        return None
    r = aid // 8
    c = aid % 8
    return (r, c)