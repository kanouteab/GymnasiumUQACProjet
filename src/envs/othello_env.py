# src/envs/othello_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

'''
─────────────────────────────────────────────────────────────────
Représentation : BITBOARDS

Board = Tuple[int, int] = (black_bb, white_bb)
  Bit i (0 = LSB) ↔ case (row = i // 8, col = i % 8)
  Exemple : case (0,0) = bit 0, case (0,7) = bit 7,
            case (7,7) = bit 63.

Trois états encodés implicitement :
  black_bb[i]=1, white_bb[i]=0  →  Noir
  black_bb[i]=0, white_bb[i]=1  →  Blanc
  black_bb[i]=0, white_bb[i]=0  →  Vide
  (11 interdit, invariant garanti par apply_move)

player : +1 = Noir, -1 = Blanc
─────────────────────────────────────────────────────────────────
'''

BOARD_SIZE = 8
FULL: int = 0xFFFF_FFFF_FFFF_FFFF   # 64 bits à 1

# Masques colonnes (pour éviter le wrap-around lors des shifts)
#   NOT_A_FILE : interdit colonne 0 avant un shift vers l'Ouest
#   NOT_H_FILE : interdit colonne 7 avant un shift vers l'Est
NOT_A_FILE: int = 0xFEFE_FEFE_FEFE_FEFE   # col != 0
NOT_H_FILE: int = 0x7F7F_7F7F_7F7F_7F7F   # col != 7

# ── 8 directions sous forme (décalage, masque_colonne) ──────────
# décalage > 0 → left-shift (<<), < 0 → right-shift (>>)
# Vérification :
#   E  : bit+1 = col+1          → <<1, masquer col 7
#   W  : bit-1 = col-1          → >>1, masquer col 0
#   S  : bit+8 = row+1          → <<8, pas de masque
#   N  : bit-8 = row-1          → >>8, pas de masque
#   SE : bit+9 = row+1,col+1    → <<9, masquer col 7
#   SW : bit+7 = row+1,col-1    → <<7, masquer col 0
#   NE : bit-7 = row-1,col+1    → >>7, masquer col 7
#   NW : bit-9 = row-1,col-1    → >>9, masquer col 0

_DIRECTIONS: List[Tuple[int, int]] = [
    ( 8, FULL),        # S
    (-8, FULL),        # N
    ( 1, NOT_H_FILE),  # E
    (-1, NOT_A_FILE),  # W
    ( 9, NOT_H_FILE),  # SE
    ( 7, NOT_A_FILE),  # SW
    (-7, NOT_H_FILE),  # NE
    (-9, NOT_A_FILE),  # NW
]

# ─────────────────────────────────────────────────────────────────
# Primitives bitboard  (fonctions internes)
# ─────────────────────────────────────────────────────────────────

def _shift(bb: int, d: int, mask: int) -> int:
    """Applique le masque puis le décalage dans une direction."""
    bb &= mask
    if d > 0:
        return (bb << d) & FULL
    return bb >> (-d)


def _popcount(bb: int) -> int:
    """Nombre de bits à 1."""
    return bin(bb).count('1')


def _legal_moves_bb(my_bb: int, opp_bb: int) -> int:
    """
    Retourne un bitboard de tous les coups légaux pour le joueur
    dont les pions sont dans my_bb.

    Algorithme Dumb-7 fill :
      Pour chaque direction, on propage depuis NOS pions à travers
      les pions ADVERSES (flood). Si la case suivante est vide,
      c'est un coup légal.
      6 étapes de propagation suffisent (max. 6 pions adverses
      en ligne sur un plateau 8×8).
    """
    empty = (~(my_bb | opp_bb)) & FULL
    legal = 0
    for d, mask in _DIRECTIONS:
        flood = _shift(my_bb, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        legal |= _shift(flood, d, mask) & empty
    return legal


def _apply_move_bb(my_bb: int, opp_bb: int, move_bit: int) -> Tuple[int, int]:
    """
    Applique un coup (move_bit = 1 << sq) et retourne (new_my, new_opp).

    Même flood que pour les coups légaux, mais cette fois on vérifie
    que le flood se termine sur un de MES pions. Si oui, les pions
    du flood sont retournés.
    """
    flipped = 0
    for d, mask in _DIRECTIONS:
        flood = _shift(move_bit, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        flood |= _shift(flood, d, mask) & opp_bb
        # Le flood est valide seulement s'il se termine sur un de nos pions
        if _shift(flood, d, mask) & my_bb:
            flipped |= flood
    new_my  = (my_bb  | move_bit | flipped)
    new_opp = (opp_bb & ~flipped)
    return new_my, new_opp


# ─────────────────────────────────────────────────────────────────
# API publique  (même interface qu'avant pour le reste du projet)
# ─────────────────────────────────────────────────────────────────

Board = Tuple[int, int]  # (black_bb, white_bb)
Move  = Tuple[int, int]  # (row, col)


def _player_boards(board: Board, player: int) -> Tuple[int, int]:
    """Retourne (my_bb, opp_bb) selon player."""
    black_bb, white_bb = board
    if player == 1:
        return black_bb, white_bb
    return white_bb, black_bb


def initial_board() -> Board:
    """Plateau initial d'Othello 8×8."""
    # Noir  : (3,4)→sq28, (4,3)→sq35
    # Blanc : (3,3)→sq27, (4,4)→sq36
    black_bb = (1 << 28) | (1 << 35)
    white_bb = (1 << 27) | (1 << 36)
    return (black_bb, white_bb)


def get_legal_moves(board: Board, player: int) -> List[Move]:
    """Liste de tous les coups légaux (row, col) pour player."""
    my_bb, opp_bb = _player_boards(board, player)
    legal_bb = _legal_moves_bb(my_bb, opp_bb)
    moves: List[Move] = []
    bb = legal_bb
    while bb:
        lsb = bb & (-bb)               # bit le plus bas isolé
        sq  = lsb.bit_length() - 1     # indice de ce bit
        moves.append((sq >> 3, sq & 7))  # (sq//8, sq%8)
        bb ^= lsb                      # retirer ce bit
    return moves


def apply_move(board: Board, player: int, move: Move) -> Board:
    """Applique un coup légal (row, col) pour player et retourne le nouveau board."""
    my_bb, opp_bb = _player_boards(board, player)
    x, y = move
    move_bit = 1 << (x * 8 + y)
    new_my, new_opp = _apply_move_bb(my_bb, opp_bb, move_bit)
    if player == 1:
        return (new_my, new_opp)
    return (new_opp, new_my)


def has_any_legal_move(board: Board, player: int) -> bool:
    my_bb, opp_bb = _player_boards(board, player)
    return _legal_moves_bb(my_bb, opp_bb) != 0


def is_terminal(board: Board) -> bool:
    """
    Terminal si le plateau est plein ou si aucun joueur n'a de coup légal.
    """
    black_bb, white_bb = board
    if (black_bb | white_bb) == FULL:
        return True
    return not has_any_legal_move(board, 1) and not has_any_legal_move(board, -1)


def score(board: Board) -> int:
    """Score final : #Noirs − #Blancs."""
    black_bb, white_bb = board
    return _popcount(black_bb) - _popcount(white_bb)


def get_winner(board: Board) -> int:
    """Retourne +1 (Noir), -1 (Blanc) ou 0 (nul)."""
    s = score(board)
    if s > 0: return 1
    if s < 0: return -1
    return 0


# ─────────────────────────────────────────────────────────────────
# Utilitaires de conversion  (affichage, Gymnasium observation)
# ─────────────────────────────────────────────────────────────────

def board_to_array(board: Board) -> np.ndarray:
    """Convertit un bitboard en np.ndarray (8,8) int8 (-1, 0, +1)."""
    black_bb, white_bb = board
    arr = np.zeros(64, dtype=np.int8)
    bb = black_bb
    while bb:
        lsb = bb & (-bb)
        arr[lsb.bit_length() - 1] = 1
        bb ^= lsb
    bb = white_bb
    while bb:
        lsb = bb & (-bb)
        arr[lsb.bit_length() - 1] = -1
        bb ^= lsb
    return arr.reshape(8, 8)


def encode_action(move: Optional[Move]) -> int:
    """Encode un move (row,col) → entier [0..63]. PASS → 64."""
    if move is None:
        return 64
    x, y = move
    return x * BOARD_SIZE + y


def decode_action(a: int) -> Optional[Move]:
    """Décode un entier [0..64] → move (row,col). 64 → PASS (None)."""
    if a == 64:
        return None
    return (a // BOARD_SIZE, a % BOARD_SIZE)


# ─────────────────────────────────────────────────────────────────
# (Optionnel) Gymnasium Env
# ─────────────────────────────────────────────────────────────────

@dataclass
class OthelloState:
    board: Board
    player: int


class OthelloEnv(gym.Env if gym is not None else object):
    """
    Env Gymnasium minimal pour Othello (bitboards).
    - action_space : Discrete(65) — 64 cases + PASS=64
    - observation  : board 8×8 (int8, -1/0/+1) via board_to_array()
    - reward       : terminal seulement (+1/-1/0 du point de vue Noir)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        if gym is None:
            raise ImportError("gymnasium is not installed. pip install gymnasium")
        super().__init__()
        self.action_space      = spaces.Discrete(65)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.int8)
        self.state      = OthelloState(board=initial_board(), player=1)
        self.done       = False
        self.render_mode = render_mode

    def _obs(self) -> np.ndarray:
        return board_to_array(self.state.board)

    def legal_actions(self) -> List[int]:
        moves = get_legal_moves(self.state.board, self.state.player)
        if moves:
            return [encode_action(m) for m in moves]
        return [64]  # PASS forcé

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = OthelloState(board=initial_board(), player=1)
        self.done  = False
        return self._obs(), {"player": self.state.player}

    def step(self, action: int):
        if self.done:
            return self._obs(), 0.0, True, False, {"already_done": True}

        if not self.action_space.contains(action):
            self.done = True
            return self._obs(), -1.0, True, False, {"invalid_action": True}

        legal = self.legal_actions()
        if action not in legal:
            self.done = True
            r = -1.0 if self.state.player == 1 else 1.0
            return self._obs(), r, True, False, {"illegal_move": True, "legal_actions": legal}

        move = decode_action(action)
        if move is None:
            self.state.player *= -1   # PASS
        else:
            self.state.board   = apply_move(self.state.board, self.state.player, move)
            self.state.player *= -1

        if is_terminal(self.state.board):
            self.done = True
            w = get_winner(self.state.board)
            return self._obs(), float(w), True, False, {"winner": w, "score": score(self.state.board)}

        return self._obs(), 0.0, False, False, {"player": self.state.player}

    def render(self):
        b = board_to_array(self.state.board)
        print("Player to move:", "Noir(1)" if self.state.player == 1 else "Blanc(-1)")
        for i in range(8):
            row = []
            for j in range(8):
                v = b[i, j]
                row.append("N" if v == 1 else ("B" if v == -1 else "."))
            print(" ".join(row))
        print("Score (#N - #B):", score(self.state.board))
        print("Legal moves:", [decode_action(a) for a in self.legal_actions()])
        print()


# ─────────────────────────────────────────────────────────────────
# Auto-test
# ─────────────────────────────────────────────────────────────────

def _self_test():
    b = initial_board()
    lm = get_legal_moves(b, 1)
    assert len(lm) == 4, f"Expected 4 initial legal moves for black, got {len(lm)}"
    nb = apply_move(b, 1, lm[0])
    assert isinstance(nb, tuple) and len(nb) == 2
    assert _popcount(nb[0]) + _popcount(nb[1]) > _popcount(b[0]) + _popcount(b[1])
    print("Self-test passed!")

if __name__ == "__main__":
    _self_test()
    if gym is not None:
        env = OthelloEnv()
        obs, info = env.reset()
        env.render()
