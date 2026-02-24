# src/agents/mcts.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.envs.othello_env import (
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
    Board,
)

Move = Tuple[int, int]  # (x,y) in [0..7]


def _value_from_player_perspective(winner: int, player: int) -> float:
    """
    winner: +1 noir, -1 blanc, 0 nul
    player: joueur pour lequel on évalue (+1/-1)
    return: +1 si player gagne, -1 si player perd, 0 nul
    """
    if winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0


@dataclass
class Node:
    board: Board                        # (black_bb, white_bb) tuple immuable
    player_to_move: int                 # +1 noir, -1 blanc
    parent: Optional["Node"] = None
    parent_move: Optional[Optional[Move]] = None  # Move or None(pass)
    children: Dict[Optional[Move], "Node"] = field(default_factory=dict)

    untried_moves: List[Optional[Move]] = field(default_factory=list)

    # stats: valeur du point de vue du joueur qui doit jouer à CE noeud
    N: int = 0
    W: float = 0.0

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTSAgent:
    """
    MCTS (UCT) pour Othello avec REUSE d'arbre (re-rooting).

    Valeurs stockées:
    - Chaque noeud stocke Q du point de vue de node.player_to_move
      (donc au backprop on inverse le signe à chaque niveau).

    UCT (du point de vue du parent):
    - Quand le parent choisit un enfant, c'est l'adversaire qui est "to move" dans l'enfant,
      donc l'exploitation pour le parent est (-child.Q()).
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

        # Pour le reuse
        self.root: Optional[Node] = None

    # -----------------------
    # Reuse API
    # -----------------------
    def reset_tree(self) -> None:
        """Appeler au début d'une nouvelle partie (ou recréer l'agent)."""
        self.root = None

    def observe_move(self, move: Optional[Move], next_board: Board, next_player_to_move: int) -> None:
        """
        Informe l'agent d'un coup joué (par lui ou l'adversaire) pour re-rooter l'arbre.

        move: (x,y) ou None (PASS)
        next_board: board APRÈS le coup
        next_player_to_move: joueur qui doit jouer ensuite
        """
        if self.root is None:
            return

        child = self.root.children.get(move)
        if child is not None and child.board == next_board and child.player_to_move == next_player_to_move:
            # Re-root: on garde le sous-arbre
            self.root = child
            self.root.parent = None
            self.root.parent_move = None
            return

        # sinon on perd l'alignement (l'adversaire a joué un coup hors arbre, ou arbre trop petit)
        self.root = None

    # -----------------------
    # Public API
    # -----------------------
    def select_move(self, board: Board, player_to_move: int) -> Optional[Move]:
        """
        Retourne le meilleur coup (x,y) ou None pour PASS (si aucun coup légal).
        """
        legal = get_legal_moves(board, player_to_move)
        if not legal:
            return None  # PASS forced

        # Reuse si possible
        if self.root is not None and self.root.board == board and self.root.player_to_move == player_to_move:
            root = self.root
        else:
            root = self._make_root(board, player_to_move)
            self.root = root

        for _ in range(self.n_simulations):
            leaf = self._select(root)
            value = self._simulate_from(leaf)  # valeur du point de vue leaf.player_to_move
            self._backpropagate(leaf, value)

        # Choix final: enfant le plus visité (robuste)
        best_move, _best_child = max(root.children.items(), key=lambda kv: kv[1].N)
        return best_move

    # -----------------------
    # Core MCTS phases
    # -----------------------
    def _make_root(self, board: Board, player: int) -> Node:
        root = Node(board=board, player_to_move=player)
        root.untried_moves = self._legal_moves_with_pass(root.board, root.player_to_move)
        return root

    def _legal_moves_with_pass(self, board: Board, player: int) -> List[Optional[Move]]:
        moves = get_legal_moves(board, player)
        if moves:
            return list(moves)
        return [None]  # PASS allowed/forced

    def _uct_score(self, parent: Node, child: Node) -> float:
        if child.N == 0:
            return float("inf")

        # exploitation du point de vue du parent (parent joue, puis enfant = adversaire)
        exploitation = -child.Q()
        exploration = self.c * math.sqrt(math.log(parent.N) / child.N)
        return exploitation + exploration

    def _select(self, node: Node) -> Node:
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

        if move is None:
            next_board = node.board
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

    def _simulate_from(self, node: Node) -> float:
        """
        Rollout random jusqu'au terminal (ou max steps).
        Retour: valeur du point de vue de node.player_to_move (au départ du rollout).
        """
        board = node.board
        player = node.player_to_move
        start_player = player

        for _ in range(self.rollout_max_steps):
            if is_terminal(board):
                break

            moves = get_legal_moves(board, player)
            if not moves:
                player = -player
                continue

            move = self.rng.choice(moves)
            board = apply_move(board, player, move)
            player = -player

        if is_terminal(board):
            w = get_winner(board)
            return _value_from_player_perspective(w, start_player)

        # cutoff
        if not self.use_score_rollout_tiebreak:
            return 0.0

        # tie-break léger via score relatif
        s = score(board)
        if s == 0:
            return 0.0
        approx = max(-1.0, min(1.0, s / 64.0))
        # score est du point de vue Noir: convertir au point de vue start_player
        return approx if start_player == 1 else -approx

    def _backpropagate(self, node: Node, value: float) -> None:
        """
        Backprop avec inversion de signe à chaque parent:
        - node stocke valeur du point de vue node.player_to_move
        - parent stocke valeur du point de vue parent.player_to_move (= - value)
        """
        current = node
        v = value
        while current is not None:
            current.N += 1
            current.W += v
            v = -v
            current = current.parent
