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

# Numba est requis — installer via : pip install numba
try:
    import numba
except ImportError as e:
    raise ImportError(
        "Numba est requis.\n"
        "Installer via : pip install numba"
    ) from e

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
NOT_A_FILE: int = 0xFEFE_FEFE_FEFE_FEFE   # col != 0  (used for W/SW/NW)
NOT_H_FILE: int = 0x7F7F_7F7F_7F7F_7F7F   # col != 7  (used for E/SE/NE)

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
# Kernels JIT Numba  (uint64, directions déroulées explicitement)
#
#   @njit n'accepte pas les boucles sur des listes Python.
#   Les 8 directions sont donc écrites en clair — le compilateur
#   LLVM peut en plus les vectoriser / pipeline automatiquement.
#
#   Numba exige des types à largeur fixe. uint64 wrappe naturellement
#   à 2^64 (même comportement qu'un registre CPU 64-bit), ce qui est
#   exactement ce qu'on veut pour un bitboard 64 cases.
#
# cache=True
#   Le bytecode compilé LLVM est sauvegardé sur disque dans __pycache__.
#   Le "warm-up" (~2s) n'a lieu qu'une seule fois ; les runs suivants
#   chargent directement le binaire.
# ─────────────────────────────────────────────────────────────────

@numba.njit(
    numba.int64(numba.uint64),
    cache=True,
)
def _popcount_nb(bb: np.uint64) -> np.int64:
    """
    Algorithme de Brian Kernighan : bb & (bb-1) efface le bit le plus bas.
    Itère seulement autant de fois qu'il y a de bits à 1 (max 64).
    bin().count('1') n'est pas supporté par Numba.
    """
    count = np.int64(0)
    while bb:
        bb = bb & (bb - np.uint64(1))
        count += np.int64(1)
    return count

@numba.njit(
    numba.uint64(numba.uint64, numba.uint64),
    cache=True,
)
def _legal_moves_nb(my: np.uint64, opp: np.uint64) -> np.uint64:
    """
    Dumb-7 fill JIT : coups légaux pour le joueur 'my' contre 'opp'.

    Pour chaque direction :
      1. On masque les bits du flood avant de shifter (évite wrap-around).
      2. On propage à travers les pions adverses (& opp), 6 fois max.
      3. La case après le flood doit être vide → coup légal.

    Chaque bloc de ~8 lignes = une direction. LLVM les compile
    en code machine natif sans interpréteur Python.
    """
    NOT_H = np.uint64(0x7F7F_7F7F_7F7F_7F7F)
    NOT_A = np.uint64(0xFEFE_FEFE_FEFE_FEFE)
    empty  = ~(my | opp)
    legal  = np.uint64(0)

    # ── S (+8) — pas de masque colonne ───────────────────────
    f  = (my << np.uint64(8)) & opp
    f |= (f  << np.uint64(8)) & opp
    f |= (f  << np.uint64(8)) & opp
    f |= (f  << np.uint64(8)) & opp
    f |= (f  << np.uint64(8)) & opp
    f |= (f  << np.uint64(8)) & opp
    legal |= (f << np.uint64(8)) & empty

    # ── N (-8) — pas de masque colonne ───────────────────────
    f  = (my >> np.uint64(8)) & opp
    f |= (f  >> np.uint64(8)) & opp
    f |= (f  >> np.uint64(8)) & opp
    f |= (f  >> np.uint64(8)) & opp
    f |= (f  >> np.uint64(8)) & opp
    f |= (f  >> np.uint64(8)) & opp
    legal |= (f >> np.uint64(8)) & empty

    # ── E (+1) — masque NOT_H : interdit col 7 avant <<1 ─────
    f  = ((my & NOT_H) << np.uint64(1)) & opp
    f |= ((f  & NOT_H) << np.uint64(1)) & opp
    f |= ((f  & NOT_H) << np.uint64(1)) & opp
    f |= ((f  & NOT_H) << np.uint64(1)) & opp
    f |= ((f  & NOT_H) << np.uint64(1)) & opp
    f |= ((f  & NOT_H) << np.uint64(1)) & opp
    legal |= ((f & NOT_H) << np.uint64(1)) & empty

    # ── W (-1) — masque NOT_A : interdit col 0 avant >>1 ─────
    f  = ((my & NOT_A) >> np.uint64(1)) & opp
    f |= ((f  & NOT_A) >> np.uint64(1)) & opp
    f |= ((f  & NOT_A) >> np.uint64(1)) & opp
    f |= ((f  & NOT_A) >> np.uint64(1)) & opp
    f |= ((f  & NOT_A) >> np.uint64(1)) & opp
    f |= ((f  & NOT_A) >> np.uint64(1)) & opp
    legal |= ((f & NOT_A) >> np.uint64(1)) & empty

    # ── SE (+9) — masque NOT_H ────────────────────────────────
    f  = ((my & NOT_H) << np.uint64(9)) & opp
    f |= ((f  & NOT_H) << np.uint64(9)) & opp
    f |= ((f  & NOT_H) << np.uint64(9)) & opp
    f |= ((f  & NOT_H) << np.uint64(9)) & opp
    f |= ((f  & NOT_H) << np.uint64(9)) & opp
    f |= ((f  & NOT_H) << np.uint64(9)) & opp
    legal |= ((f & NOT_H) << np.uint64(9)) & empty

    # ── SW (+7) — masque NOT_A ────────────────────────────────
    f  = ((my & NOT_A) << np.uint64(7)) & opp
    f |= ((f  & NOT_A) << np.uint64(7)) & opp
    f |= ((f  & NOT_A) << np.uint64(7)) & opp
    f |= ((f  & NOT_A) << np.uint64(7)) & opp
    f |= ((f  & NOT_A) << np.uint64(7)) & opp
    f |= ((f  & NOT_A) << np.uint64(7)) & opp
    legal |= ((f & NOT_A) << np.uint64(7)) & empty

    # ── NE (-7) — masque NOT_H ────────────────────────────────
    f  = ((my & NOT_H) >> np.uint64(7)) & opp
    f |= ((f  & NOT_H) >> np.uint64(7)) & opp
    f |= ((f  & NOT_H) >> np.uint64(7)) & opp
    f |= ((f  & NOT_H) >> np.uint64(7)) & opp
    f |= ((f  & NOT_H) >> np.uint64(7)) & opp
    f |= ((f  & NOT_H) >> np.uint64(7)) & opp
    legal |= ((f & NOT_H) >> np.uint64(7)) & empty

    # ── NW (-9) — masque NOT_A ────────────────────────────────
    f  = ((my & NOT_A) >> np.uint64(9)) & opp
    f |= ((f  & NOT_A) >> np.uint64(9)) & opp
    f |= ((f  & NOT_A) >> np.uint64(9)) & opp
    f |= ((f  & NOT_A) >> np.uint64(9)) & opp
    f |= ((f  & NOT_A) >> np.uint64(9)) & opp
    f |= ((f  & NOT_A) >> np.uint64(9)) & opp
    legal |= ((f & NOT_A) >> np.uint64(9)) & empty

    return legal

@numba.njit(
    numba.types.UniTuple(numba.uint64, 2)(numba.uint64, numba.uint64, numba.uint64),
    cache=True,
)
def _apply_move_nb(
    my: np.uint64, opp: np.uint64, move_bit: np.uint64
) -> Tuple[np.uint64, np.uint64]:
    """
    Même logique de flood que _legal_moves_nb, mais depuis la case jouée.
    Un flood dans une direction est valide (→ capture) seulement si
    la case APRÈS le flood appartient à notre joueur (my & next_step != 0).
    """
    NOT_H   = np.uint64(0x7F7F_7F7F_7F7F_7F7F)
    NOT_A   = np.uint64(0xFEFE_FEFE_FEFE_FEFE)
    # Note : on évite |= dans les branches conditionnelles.
    # Numba (certaines versions) perd le type uint64 avec l'affectation
    # augmentée en SSA → on utilise flipped = flipped | f explicitement.
    flipped = np.uint64(0)

    # ── S (+8) ──────────────────────────────────────────────
    f  = (move_bit << np.uint64(8)) & opp
    f  = f | ((f << np.uint64(8)) & opp)
    f  = f | ((f << np.uint64(8)) & opp)
    f  = f | ((f << np.uint64(8)) & opp)
    f  = f | ((f << np.uint64(8)) & opp)
    f  = f | ((f << np.uint64(8)) & opp)
    if (f << np.uint64(8)) & my:
        flipped = flipped | f

    # ── N (−8) ──────────────────────────────────────────────
    f  = (move_bit >> np.uint64(8)) & opp
    f  = f | ((f >> np.uint64(8)) & opp)
    f  = f | ((f >> np.uint64(8)) & opp)
    f  = f | ((f >> np.uint64(8)) & opp)
    f  = f | ((f >> np.uint64(8)) & opp)
    f  = f | ((f >> np.uint64(8)) & opp)
    if (f >> np.uint64(8)) & my:
        flipped = flipped | f

    # ── E (+1, masque NOT_H) ─────────────────────────────────
    f  = ((move_bit & NOT_H) << np.uint64(1)) & opp
    f  = f | (((f & NOT_H) << np.uint64(1)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(1)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(1)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(1)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(1)) & opp)
    if ((f & NOT_H) << np.uint64(1)) & my:
        flipped = flipped | f

    # ── W (−1, masque NOT_A) ─────────────────────────────────
    f  = ((move_bit & NOT_A) >> np.uint64(1)) & opp
    f  = f | (((f & NOT_A) >> np.uint64(1)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(1)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(1)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(1)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(1)) & opp)
    if ((f & NOT_A) >> np.uint64(1)) & my:
        flipped = flipped | f

    # ── SE (+9, masque NOT_H) ────────────────────────────────
    f  = ((move_bit & NOT_H) << np.uint64(9)) & opp
    f  = f | (((f & NOT_H) << np.uint64(9)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(9)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(9)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(9)) & opp)
    f  = f | (((f & NOT_H) << np.uint64(9)) & opp)
    if ((f & NOT_H) << np.uint64(9)) & my:
        flipped = flipped | f

    # ── SW (+7, masque NOT_A) ────────────────────────────────
    f  = ((move_bit & NOT_A) << np.uint64(7)) & opp
    f  = f | (((f & NOT_A) << np.uint64(7)) & opp)
    f  = f | (((f & NOT_A) << np.uint64(7)) & opp)
    f  = f | (((f & NOT_A) << np.uint64(7)) & opp)
    f  = f | (((f & NOT_A) << np.uint64(7)) & opp)
    f  = f | (((f & NOT_A) << np.uint64(7)) & opp)
    if ((f & NOT_A) << np.uint64(7)) & my:
        flipped = flipped | f

    # ── NE (−7, masque NOT_H) ────────────────────────────────
    f  = ((move_bit & NOT_H) >> np.uint64(7)) & opp
    f  = f | (((f & NOT_H) >> np.uint64(7)) & opp)
    f  = f | (((f & NOT_H) >> np.uint64(7)) & opp)
    f  = f | (((f & NOT_H) >> np.uint64(7)) & opp)
    f  = f | (((f & NOT_H) >> np.uint64(7)) & opp)
    f  = f | (((f & NOT_H) >> np.uint64(7)) & opp)
    if ((f & NOT_H) >> np.uint64(7)) & my:
        flipped = flipped | f

    # ── NW (−9, masque NOT_A) ────────────────────────────────
    f  = ((move_bit & NOT_A) >> np.uint64(9)) & opp
    f  = f | (((f & NOT_A) >> np.uint64(9)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(9)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(9)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(9)) & opp)
    f  = f | (((f & NOT_A) >> np.uint64(9)) & opp)
    if ((f & NOT_A) >> np.uint64(9)) & my:
        flipped = flipped | f

    new_my  = my  | move_bit | flipped
    new_opp = opp & ~flipped
    return new_my, new_opp

# ── Wrappers : pass-through directs, zéro conversion ────────────
# Board stocke maintenant np.uint64 nativement : les kernels JIT
# reçoivent et retournent np.uint64 sans aucune boxing/unboxing.
def _popcount(bb: np.uint64) -> int:
    return int(_popcount_nb(bb))

def _legal_moves_bb(my_bb: np.uint64, opp_bb: np.uint64) -> np.uint64:
    return _legal_moves_nb(my_bb, opp_bb)

def _apply_move_bb(
    my_bb: np.uint64, opp_bb: np.uint64, move_bit: np.uint64
) -> Tuple[np.uint64, np.uint64]:
    return _apply_move_nb(my_bb, opp_bb, move_bit)


# ─────────────────────────────────────────────────────────────────
# API publique  (même interface qu'avant pour le reste du projet)
# ─────────────────────────────────────────────────────────────────

# Board est maintenant Tuple[np.uint64, np.uint64].
# Les agents reçoivent et retournent ce type directement :
# plus aucune conversion à la frontière JIT.
Board = Tuple[np.uint64, np.uint64]  # (black_bb, white_bb)
Move  = Tuple[int, int]              # (row, col) — inchangé


def _player_boards(board: Board, player: int) -> Tuple[np.uint64, np.uint64]:
    """Retourne (my_bb, opp_bb) selon player."""
    black_bb, white_bb = board
    if player == 1:
        return black_bb, white_bb
    return white_bb, black_bb


def initial_board() -> Board:
    """Plateau initial d'Othello 8×8."""
    # Noir  : (3,4)→sq28, (4,3)→sq35
    # Blanc : (3,3)→sq27, (4,4)→sq36
    # np.uint64 dès la construction : aucune conversion jamais nécessaire.
    black_bb = np.uint64((1 << 28) | (1 << 35))
    white_bb = np.uint64((1 << 27) | (1 << 36))
    return (black_bb, white_bb)


def get_legal_moves(board: Board, player: int) -> List[Move]:
    """Liste de tous les coups légaux (row, col) pour player."""
    my_bb, opp_bb = _player_boards(board, player)
    legal_bb = _legal_moves_bb(my_bb, opp_bb)  # np.uint64
    moves: List[Move] = []
    bb = legal_bb
    while bb:
        lsb = bb & (-bb)                    # np.uint64 : -bb wrappe en uint64 (deux's complement)
        sq  = int(lsb).bit_length() - 1     # int() requis : np.uint64 n'a pas .bit_length()
        moves.append((sq >> 3, sq & 7))     # (sq//8, sq%8)
        bb ^= lsb                           # retire ce bit
    return moves


def apply_move(board: Board, player: int, move: Move) -> Board:
    """Applique un coup légal (row, col) pour player et retourne le nouveau board."""
    my_bb, opp_bb = _player_boards(board, player)
    x, y = move
    # np.uint64 natif : aucune conversion avant d'appeler le kernel JIT
    move_bit = np.uint64(1) << np.uint64(x * 8 + y)
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
        arr[int(lsb).bit_length() - 1] = 1
        bb ^= lsb
    bb = white_bb
    while bb:
        lsb = bb & (-bb)
        arr[int(lsb).bit_length() - 1] = -1
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
