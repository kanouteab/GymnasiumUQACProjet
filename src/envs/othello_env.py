
# src/envs/othello_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None


# Convention:
# board: np.ndarray shape (8,8), dtype=int8
# 0 = vide,  1 = Noir,  -1 = Blanc
# player: 1 (Noir) ou -1 (Blanc)
BOARD_SIZE = 8

# 8 directions (dx, dy)
DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def initial_board() -> np.ndarray:
    """Plateau initial d'Othello 8x8."""
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    # centre:
    # (3,3)=Blanc, (4,4)=Blanc, (3,4)=Noir, (4,3)=Noir (convention classique)
    b[3, 3] = -1
    b[4, 4] = -1
    b[3, 4] = 1
    b[4, 3] = 1
    return b


def on_board(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _captures_in_dir(board: np.ndarray, player: int, x: int, y: int, dx: int, dy: int) -> List[Tuple[int, int]]:
    """
    Retourne la liste des pions adverses capturés dans la direction (dx,dy)
    si on joue (x,y). Sinon [].
    """
    opp = -player
    cx, cy = x + dx, y + dy
    captured = []

    # 1) première case : adversaire
    if not on_board(cx, cy) or board[cx, cy] != opp:
        return []

    # 2) accumuler les adversaires
    while on_board(cx, cy) and board[cx, cy] == opp:
        captured.append((cx, cy))
        cx += dx
        cy += dy

    # 3) pour capturer, il faut terminer par un pion du joueur
    if not on_board(cx, cy):
        return []
    if board[cx, cy] != player:
        return []

    return captured


def get_captures(board: np.ndarray, player: int, move: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Retourne toutes les cases capturées si player joue move=(x,y). Sinon [].
    """
    x, y = move
    if not on_board(x, y) or board[x, y] != 0:
        return []
    caps: List[Tuple[int, int]] = []
    for dx, dy in DIRS:
        caps.extend(_captures_in_dir(board, player, x, y, dx, dy))
    return caps


def is_legal_move(board: np.ndarray, player: int, move: Tuple[int, int]) -> bool:
    return len(get_captures(board, player, move)) > 0


def get_legal_moves(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """
    Liste de tous les coups légaux (x,y) pour player.
    """
    empties = np.argwhere(board == 0)
    moves: List[Tuple[int, int]] = []
    for x, y in empties:
        if is_legal_move(board, player, (int(x), int(y))):
            moves.append((int(x), int(y)))
    return moves


def apply_move(board: np.ndarray, player: int, move: Tuple[int, int]) -> np.ndarray:
    """
    Applique un coup légal (x,y) pour player et renvoie un nouveau board.
    Lève ValueError si le coup est illégal.
    """
    caps = get_captures(board, player, move)
    if not caps:
        raise ValueError(f"Illegal move {move} for player {player}")

    nb = board.copy()
    x, y = move
    nb[x, y] = player
    for cx, cy in caps:
        nb[cx, cy] = player
    return nb


def has_any_legal_move(board: np.ndarray, player: int) -> bool:
    return len(get_legal_moves(board, player)) > 0


def is_terminal(board: np.ndarray) -> bool:
    """
    Terminal si:
    - plateau plein, ou
    - aucun des deux joueurs n'a de coup légal
    """
    if np.all(board != 0):
        return True
    if not has_any_legal_move(board, 1) and not has_any_legal_move(board, -1):
        return True
    return False


def score(board: np.ndarray) -> int:
    """
    Score final simple: (#Noirs - #Blancs)
    """
    return int(np.sum(board))


def get_winner(board: np.ndarray) -> int:
    """
    Renvoie:
    1  si Noir gagne,
    -1 si Blanc gagne,
    0  si nul.
    (Sur plateau terminal de préférence.)
    """
    s = score(board)
    if s > 0:
        return 1
    if s < 0:
        return -1
    return 0


def encode_action(move: Optional[Tuple[int, int]]) -> int:
    """
    Encode un move (x,y) vers un entier [0..63]. Le pass est 64.
    """
    if move is None:
        return 64
    x, y = move
    return x * BOARD_SIZE + y


def decode_action(a: int) -> Optional[Tuple[int, int]]:
    """
    Decode un entier [0..64] vers move (x,y). 64 => pass(None)
    """
    if a == 64:
        return None
    x = a // BOARD_SIZE
    y = a % BOARD_SIZE
    return (int(x), int(y))


# (Optionnel) Gymnasium Env

@dataclass
class OthelloState:
    board: np.ndarray
    player: int


class OthelloEnv(gym.Env if gym is not None else object):
    """
    Env Gymnasium minimal pour Othello:
    - action_space: Discrete(65) (64 cases + PASS=64)
    - observation: board 8x8 (int8 -1,0,1)
    - reward: terminal seulement (+1/-1/0 du point de vue Noir)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        if gym is None:
            raise ImportError("gymnasium is not installed. pip install gymnasium")
        super().__init__()
        self.action_space = spaces.Discrete(65)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.int8)

        self.state = OthelloState(board=initial_board(), player=1)
        self.done = False
        self.render_mode = render_mode

    def legal_actions(self) -> List[int]:
        moves = get_legal_moves(self.state.board, self.state.player)
        if moves:
            return [encode_action(m) for m in moves]
        # si aucun coup légal => pass autorisé
        return [64]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = OthelloState(board=initial_board(), player=1)
        self.done = False
        info = {"player": self.state.player}
        return self.state.board.copy(), info

    def step(self, action: int):
        if self.done:
            return self.state.board.copy(), 0.0, True, False, {"already_done": True}

        if not self.action_space.contains(action):
            self.done = True
            return self.state.board.copy(), -1.0, True, False, {"invalid_action": True}

        legal = self.legal_actions()
        if action not in legal:
            # coup illégal => perdre immédiatement (du point de vue Noir)
            self.done = True
            # si le joueur courant est Noir, -1; sinon +1 (Noir gagne)
            r = -1.0 if self.state.player == 1 else 1.0
            return self.state.board.copy(), r, True, False, {"illegal_move": True, "legal_actions": legal}

        move = decode_action(action)

        # Appliquer move ou pass
        if move is None:
            # PASS: on change juste de joueur
            self.state.player *= -1
        else:
            self.state.board = apply_move(self.state.board, self.state.player, move)
            self.state.player *= -1

        # Si le joueur suivant n'a pas de coup mais l'autre en a, il doit PASS automatiquement
        # (on laisse l'agent gérer via action=64, mais ici on peut rester minimal et ne pas auto-pass)

        term = is_terminal(self.state.board)
        if term:
            self.done = True
            w = get_winner(self.state.board)
            reward = float(w)  # point de vue Noir
            return self.state.board.copy(), reward, True, False, {"winner": w, "score": score(self.state.board)}

        return self.state.board.copy(), 0.0, False, False, {"player": self.state.player}

    def render(self):
        b = self.state.board
        print("Player to move:", "Noir(1)" if self.state.player == 1 else "Blanc(-1)")
        # Affichage simple
        # N=Noir, B=Blanc, .=vide
        for i in range(8):
            row = []
            for j in range(8):
                v = b[i, j]
                row.append("N" if v == 1 else ("B" if v == -1 else "."))
            print(" ".join(row))
        print("Score (#N - #B):", score(b))
        print("Legal moves:", [decode_action(a) for a in self.legal_actions()])
        print()



# tests rapides
def _self_test():
    b = initial_board()
    # Au début, Noir a 4 coups légaux
    lm = get_legal_moves(b, 1)
    assert len(lm) == 4, f"Expected 4 initial legal moves for black, got {len(lm)}"
    # Jouer un coup légal
    nb = apply_move(b, 1, lm[0])
    assert nb.shape == (8, 8)
    assert np.sum(nb != 0) > np.sum(b != 0)

if __name__ == "__main__":
    _self_test()
    if gym is not None:
        env = OthelloEnv()
        obs, info = env.reset()
        env.render()
