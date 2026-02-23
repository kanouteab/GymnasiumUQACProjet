
# src/agents/alphabeta.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.envs.othello_env import (
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
)

Move = Tuple[int, int]


# Poids positionnels simples (heuristique standard)
# Coins très précieux, cases adjacentes aux coins souvent dangereuses.
WEIGHTS = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
], dtype=np.int32)


def evaluate(board: np.ndarray, player: int) -> float:
    """
    Heuristique pour Othello du point de vue 'player' (+1 Noir, -1 Blanc).
    Combine:
      - poids positionnels
      - mobilité (diff coups légaux)
      - coins
      - (optionnel) score brut en fin de partie
    """
    # Terminal: valeur très grande
    if is_terminal(board):
        w = get_winner(board)
        if w == 0:
            return 0.0
        return 1e9 if w == player else -1e9

    # 1) positional weights
    pos = float(np.sum(WEIGHTS * board))  # du point de vue Noir
    pos = pos if player == 1 else -pos

    # 2) mobility
    my_moves = len(get_legal_moves(board, player))
    opp_moves = len(get_legal_moves(board, -player))
    mobility = 0.0
    if my_moves + opp_moves > 0:
        mobility = 100.0 * (my_moves - opp_moves) / (my_moves + opp_moves)

    # 3) corners
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    my_c = sum(1 for (x, y) in corners if board[x, y] == player)
    opp_c = sum(1 for (x, y) in corners if board[x, y] == -player)
    corner_score = 25.0 * (my_c - opp_c)

    return pos + mobility + corner_score


@dataclass
class AlphaBetaAgent:
    depth: int = 4
    use_move_ordering: bool = True

    def select_move(self, board: np.ndarray, player: int) -> Optional[Move]:
        moves = get_legal_moves(board, player)
        if not moves:
            return None  # PASS

        # optional ordering: best-first by heuristic after 1-ply apply
        if self.use_move_ordering:
            moves = self._order_moves(board, player, moves)

        best_move = None
        best_val = -math.inf

        alpha = -math.inf
        beta = math.inf

        for mv in moves:
            nb = apply_move(board, player, mv)
            val = self._alphabeta(nb, -player, self.depth - 1, alpha, beta, root_player=player)
            if val > best_val:
                best_val = val
                best_move = mv
            alpha = max(alpha, best_val)

        return best_move

    def _order_moves(self, board: np.ndarray, player: int, moves: List[Move]) -> List[Move]:
        scored = []
        for mv in moves:
            nb = apply_move(board, player, mv)
            scored.append((evaluate(nb, player), mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [mv for _, mv in scored]

    def _alphabeta(
        self,
        board: np.ndarray,
        player_to_move: int,
        depth: int,
        alpha: float,
        beta: float,
        root_player: int,
    ) -> float:
        # Terminal or depth cutoff
        if depth <= 0 or is_terminal(board):
            return evaluate(board, root_player)

        moves = get_legal_moves(board, player_to_move)

        if not moves:
            # PASS: switch player without changing board, depth decreases
            return self._alphabeta(board, -player_to_move, depth - 1, alpha, beta, root_player)

        # Ordering helps pruning
        if self.use_move_ordering:
            moves = self._order_moves(board, player_to_move, moves)

        if player_to_move == root_player:
            # Maximizing
            value = -math.inf
            for mv in moves:
                nb = apply_move(board, player_to_move, mv)
                value = max(value, self._alphabeta(nb, -player_to_move, depth - 1, alpha, beta, root_player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            # Minimizing
            value = math.inf
            for mv in moves:
                nb = apply_move(board, player_to_move, mv)
                value = min(value, self._alphabeta(nb, -player_to_move, depth - 1, alpha, beta, root_player))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
