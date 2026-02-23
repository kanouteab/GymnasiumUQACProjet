# src/agents/mcts.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.envs.othello_env import (
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
)

Move = Tuple[int, int]  # (x,y) in [0..7]


def _result_value_from_root_perspective(winner: int, root_player: int) -> float:
    """
    winner: +1 noir gagne, -1 blanc gagne, 0 nul (définition OthelloEnv)
    root_player: +1 (noir) or -1 (blanc)
    return: +1 si root gagne, -1 si root perd, 0 nul
    """
    if winner == 0:
        return 0.0
    return 1.0 if winner == root_player else -1.0


@dataclass
class Node:
    board: np.ndarray
    player_to_move: int                 # +1 noir, -1 blanc
    parent: Optional["Node"] = None
    parent_move: Optional[Optional[Move]] = None  # Move or None(pass)
    children: Dict[Optional[Move], "Node"] = field(default_factory=dict)

    # actions not yet expanded
    untried_moves: List[Optional[Move]] = field(default_factory=list)

    # stats
    N: int = 0
    W: float = 0.0  # total value from root perspective

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTSAgent:
    """
    MCTS (UCT) pour Othello.

    - action selection via UCT:
        UCT = Q + c * sqrt(ln(N_parent)/N_child)
    - rollout: random (avec PASS si aucun coup)
    - backprop: valeur du point de vue du joueur root (celui qui choisit le coup)
    """

    def __init__(
        self,
        n_simulations: int = 800,
        c_uct: float = 1.4,
        rollout_max_steps: int = 200,
        seed: Optional[int] = None,
        use_score_rollout_tiebreak: bool = False,
    ):
        self.n_simulations = int(n_simulations)
        self.c = float(c_uct)
        self.rollout_max_steps = int(rollout_max_steps)
        self.rng = random.Random(seed)
        self.use_score_rollout_tiebreak = bool(use_score_rollout_tiebreak)

 
    # Public API
    def select_move(self, board: np.ndarray, player_to_move: int) -> Optional[Move]:
        """
        Retourne le meilleur coup (x,y) ou None pour PASS (si aucun coup légal).
        """
        legal = get_legal_moves(board, player_to_move)
        if not legal:
            return None  # PASS forced

        root = self._make_root(board, player_to_move)

        for _ in range(self.n_simulations):
            leaf = self._select(root)
            value = self._simulate_from(leaf, root_player=player_to_move)
            self._backpropagate(leaf, value)

        # Choix final: enfant le plus visité (robuste)
        best_move, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
        return best_move

    # Core MCTS phases
    def _make_root(self, board: np.ndarray, player: int) -> Node:
        root = Node(board=board.copy(), player_to_move=player)
        root.untried_moves = self._legal_moves_with_pass(root.board, root.player_to_move)
        return root

    def _legal_moves_with_pass(self, board: np.ndarray, player: int) -> List[Optional[Move]]:
        moves = get_legal_moves(board, player)
        if moves:
            return list(moves)
        return [None]  # PASS allowed/forced

    def _uct_score(self, parent: Node, child: Node) -> float:
        if child.N == 0:
            return float("inf")
        return child.Q() + self.c * math.sqrt(math.log(parent.N) / child.N)

    def _select(self, node: Node) -> Node:
        """
        Selection: descendre tant que le noeud est fully expanded et non terminal.
        Sinon, expansion.
        """
        current = node
        while True:
            if is_terminal(current.board):
                return current

            if current.untried_moves:
                return self._expand(current)

            # fully expanded: select best UCT child
            _, current = max(current.children.items(), key=lambda kv: self._uct_score(current, kv[1]))

    def _expand(self, node: Node) -> Node:
        move = node.untried_moves.pop()

        # Apply move (or pass)
        if move is None:
            next_board = node.board.copy()
            next_player = -node.player_to_move
        else:
            next_board = apply_move(node.board, node.player_to_move, move)
            next_player = -node.player_to_move

        child = Node(
            board=next_board,
            player_to_move=next_player,
            parent=node,
            parent_move=move,
        )
        child.untried_moves = self._legal_moves_with_pass(child.board, child.player_to_move)

        node.children[move] = child
        return child

    def _simulate_from(self, node: Node, root_player: int) -> float:
        """
        Simulation/Rollout: jouer aléatoirement jusqu'au terminal (ou max steps).
        Retour: valeur du point de vue root_player.
        """
        board = node.board.copy()
        player = node.player_to_move

        # Rollout random
        for _ in range(self.rollout_max_steps):
            if is_terminal(board):
                break

            moves = get_legal_moves(board, player)
            if not moves:
                # PASS
                player = -player
                continue

            move = self.rng.choice(moves)
            board = apply_move(board, player, move)
            player = -player

        # Évaluation terminale (ou approximée si cutoff)
        if is_terminal(board):
            w = get_winner(board)
            return _result_value_from_root_perspective(w, root_player)

        # si cutoff: option 1 (simple) => draw
        if not self.use_score_rollout_tiebreak:
            return 0.0

        # option 2: tie-break via score relatif (petit signal)
        # score = (#Noirs - #Blancs). Convertir au point de vue root_player.
        s = score(board)
        if s == 0:
            return 0.0
        # normaliser grossièrement dans [-1,1]
        approx = max(-1.0, min(1.0, s / 64.0))
        return approx if root_player == 1 else -approx

    def _backpropagate(self, node: Node, value: float) -> None:
        """
        Backpropagate valeur du point de vue root.
        (Même value ajoutée à tous les ancêtres, car déjà en root-perspective.)
        """
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            current = current.parent
