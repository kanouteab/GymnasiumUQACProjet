
# src/agents/alphabeta.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.envs.othello_env import (
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
    Board,
)

Move = Tuple[int, int]


# ── Poids positionnels (heuristique standard Othello) ─────────────
# Mêmes valeurs qu'avant, mais rangés en tableau plat (index = sq = row*8+col)
# pour fonctionner directement avec les bitboards.
WEIGHTS_FLAT: List[int] = [
    120, -20,  20,   5,   5,  20, -20, 120,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
     20,  -5,  15,   3,   3,  15,  -5,  20,
      5,  -5,   3,   3,   3,   3,  -5,   5,
      5,  -5,   3,   3,   3,   3,  -5,   5,
     20,  -5,  15,   3,   3,  15,  -5,  20,
    -20, -40,  -5,  -5,  -5,  -5, -40, -20,
    120, -20,  20,   5,   5,  20, -20, 120,
]

# Masque des 4 coins : (0,0)=sq0, (0,7)=sq7, (7,0)=sq56, (7,7)=sq63
CORNERS_BB: int = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)


def _positional_score(my_bb: int, opp_bb: int) -> float:
    """
    Score positionnel du point de vue de my_bb.
    Itère sur les bits allumés et somme leurs poids.
    """
    pos = 0
    bb = my_bb
    while bb:
        lsb = bb & (-bb)
        pos += WEIGHTS_FLAT[lsb.bit_length() - 1]
        bb ^= lsb
    bb = opp_bb
    while bb:
        lsb = bb & (-bb)
        pos -= WEIGHTS_FLAT[lsb.bit_length() - 1]
        bb ^= lsb
    return float(pos)


def evaluate(board: Board, player: int) -> float:
    """
    Heuristique pour Othello du point de vue 'player' (+1 Noir, -1 Blanc).
    Combine :
      - poids positionnels (WEIGHTS_FLAT, par itération sur bits)
      - mobilité (diff coups légaux)
      - coins (test de bits direct)
    """
    # Terminal : valeur très grande
    if is_terminal(board):
        w = get_winner(board)
        if w == 0:
            return 0.0
        return 1e9 if w == player else -1e9

    black_bb, white_bb = board
    my_bb  = black_bb if player == 1 else white_bb
    opp_bb = white_bb if player == 1 else black_bb

    # 1) poids positionnels
    pos = _positional_score(my_bb, opp_bb)

    # 2) mobilité
    my_moves  = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, -player))
    mobility = 0.0
    if my_moves + opp_moves > 0:
        mobility = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

    # 3) coins (test de bits : CORNERS_BB & my_bb / opp_bb)
    my_c  = bin(my_bb  & CORNERS_BB).count('1')
    opp_c = bin(opp_bb & CORNERS_BB).count('1')
    corner_score = 25.0 * (my_c - opp_c)

    return pos + mobility + corner_score


@dataclass
class AlphaBetaAgent:
    depth: int = 4
    use_move_ordering: bool = True

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        moves = get_legal_moves(board, player)
        if not moves:
            return None  # PASS

        if self.use_move_ordering:
            moves = self._order_moves(board, player, moves)

        best_move = None
        best_val  = -math.inf
        alpha     = -math.inf
        beta      = math.inf

        for mv in moves:
            nb  = apply_move(board, player, mv)
            val = self._alphabeta(nb, -player, self.depth - 1, alpha, beta, root_player=player)
            if val > best_val:
                best_val  = val
                best_move = mv
            alpha = max(alpha, best_val)

        return best_move

    def _order_moves(self, board: Board, player: int, moves: List[Move]) -> List[Move]:
        scored = [(evaluate(apply_move(board, player, mv), player), mv) for mv in moves]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [mv for _, mv in scored]

    def _alphabeta(
        self,
        board: Board,
        player_to_move: int,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int,
    ) -> float:
        if depth <= 0 or is_terminal(board):
            return evaluate(board, root_player)

        moves = get_legal_moves(board, player_to_move)

        if not moves:
            # PASS : on change de joueur sans modifier le plateau
            return self._alphabeta(board, -player_to_move, depth - 1, alpha, beta, root_player)

        if self.use_move_ordering:
            moves = self._order_moves(board, player_to_move, moves)

        if player_to_move == root_player:
            value = -math.inf
            for mv in moves:
                nb    = apply_move(board, player_to_move, mv)
                value = max(value, self._alphabeta(nb, -player_to_move, depth - 1, alpha, beta, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for mv in moves:
                nb    = apply_move(board, player_to_move, mv)
                value = min(value, self._alphabeta(nb, -player_to_move, depth - 1, alpha, beta, root_player))
                beta  = min(beta, value)
                if alpha >= beta:
                    break
            return value
