# src/agents/dqn.py
"""
Deep Q-Network (DQN) agent pour Othello.

Architecture CNN :
  Entrée : (batch, 3, 8×8)
    Canal 0 : mes pions        (float32  0/1)
    Canal 1 : pions adverses   (float32  0/1)
    Canal 2 : coups légaux     (float32  0/1)
  Corps  : 3 couches Conv2d(padding=1) → features spatiales 8×8
  Tête   : Linear(8192→256) → Linear(256→65)  [64 cases + PASS]

Composants DQN :
  - Replay buffer circulaire numpy (pré-alloué, O(1) push/sample)
  - Target network — mise à jour hard toutes les `target_update_freq` steps
  - Masking des actions illégales à l'inférence ET dans le calcul de la cible
  - Huber loss (smooth_l1) + gradient clipping (norme ≤ 1)
  - Epsilon-greedy avec décroissance par épisode

Interface compatible tournament.py / othello_pygame.py :
  agent.select_move(board, player) → Optional[Move]

Entraînement (utilisé par train_dqn.py) :
  agent.board_to_obs(board, player) → np.ndarray (3,8,8)
  agent.select_action(obs, legal_ids) → int
  agent.push(obs, a, r, next_obs, done, next_legal)
  agent.update() → Optional[float]
  agent.decay_epsilon()
  agent.save(path) / agent.load(path)
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.othello_env import (
    board_to_array, decode_action, encode_action, get_legal_moves,
)

Board = Tuple[int, int]
Move  = Tuple[int, int]


# ── Réseau de neurones ─────────────────────────────────────────────────────────

class OthelloNet(nn.Module):
    """
    CNN pour estimer Q(s, a) sur Othello 8×8.

    Les couches Conv2d(padding=1) conservent la résolution 8×8 à chaque étape,
    ce qui permet au réseau d'exploiter la structure spatiale du plateau
    (coins, bords, lignes, diagonales) tout au long du traitement.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 128 filtres × 8×8 cases = 8 192 features
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 65),      # 64 cases + 1 PASS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 3, 8, 8)  →  sortie : (B, 65)"""
        return self.fc(self.conv(x).flatten(1))


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Buffer circulaire pré-alloué en numpy.

    Stocke les masques de coups légaux pour s' afin de permettre
    un masking correct lors du calcul de la valeur cible (sans biais
    vers des actions illégales dont Q ≈ 0 en début d'entraînement).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.pos      = 0
        self.size     = 0
        # Tableaux pré-alloués — aucun alloc dynamique après __init__
        self._obs        = np.zeros((capacity, 3, 8, 8), dtype=np.float32)
        self._actions    = np.zeros(capacity,            dtype=np.int64)
        self._rewards    = np.zeros(capacity,            dtype=np.float32)
        self._next_obs   = np.zeros((capacity, 3, 8, 8), dtype=np.float32)
        self._dones      = np.zeros(capacity,            dtype=np.float32)
        self._next_legal = np.zeros((capacity, 65),      dtype=np.bool_)

    def push(
        self,
        obs:        np.ndarray,
        action:     int,
        reward:     float,
        next_obs:   np.ndarray,
        done:       bool,
        next_legal: List[int],
    ) -> None:
        p = self.pos
        self._obs[p]      = obs
        self._actions[p]  = action
        self._rewards[p]  = reward
        self._next_obs[p] = next_obs
        self._dones[p]    = float(done)
        self._next_legal[p].fill(False)
        for a in next_legal:
            self._next_legal[p, a] = True
        self.pos  = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self._obs[idx]).to(device),
            torch.from_numpy(self._actions[idx]).to(device),
            torch.from_numpy(self._rewards[idx]).to(device),
            torch.from_numpy(self._next_obs[idx]).to(device),
            torch.from_numpy(self._dones[idx]).to(device),
            torch.from_numpy(self._next_legal[idx]).to(device),  # BoolTensor
        )

    def __len__(self) -> int:
        return self.size


# ── Agent ──────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Agent DQN complet pour Othello.

    Hyperparamètres par défaut calibrés pour un RTX 5060 (8 Go VRAM)
    et 5 000 épisodes d'entraînement :
      - batch_size=256          : GPU < 1 Mo par batch, très rapide
      - buffer_capacity=100 000 : ~230 Mo RAM pour les tableaux numpy
      - target_update_freq=200  : ~1 sync / 200 coups ≈ 7 parties
      - eps_decay=0.9992        : atteint eps_min=0.05 vers l'épisode 3 700
    """

    def __init__(
        self,
        lr:                 float        = 1e-4,
        gamma:              float        = 0.99,
        eps:                float        = 1.0,
        eps_min:            float        = 0.05,
        eps_decay:          float        = 0.9992,
        buffer_capacity:    int          = 100_000,
        batch_size:         int          = 256,
        target_update_freq: int          = 200,
        min_buffer_size:    int          = 1_000,
        device:             Optional[str] = None,
        seed:               int          = 0,
    ) -> None:
        self.gamma              = gamma
        self.eps                = eps
        self.eps_min            = eps_min
        self.eps_decay          = eps_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size    = min_buffer_size
        self.steps              = 0
        self.rng                = random.Random(seed)

        # Sélection du device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Réseaux online et target
        self.online_net = OthelloNet().to(self.device)
        self.target_net = OthelloNet().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

    # ── Conversion état → observation ─────────────────────────────────────────

    def board_to_obs(self, board: Board, player: int) -> np.ndarray:
        """
        Encode un bitboard en observation 3 canaux (3, 8, 8) float32.
        Toujours du point de vue de `player` (mes pions = canal 0 = +1).

        Canal 0 : mes pions        — 1.0 là où j'ai un pion
        Canal 1 : pions adverses   — 1.0 là où l'adversaire a un pion
        Canal 2 : coups légaux     — 1.0 sur les cases où je peux jouer
        """
        arr = board_to_array(board)          # (8,8) int8, black=+1, white=-1
        if player == -1:
            arr = -arr                       # flip : je deviens +1

        my    = (arr > 0).astype(np.float32)
        opp   = (arr < 0).astype(np.float32)
        legal = np.zeros((8, 8), dtype=np.float32)
        for r, c in get_legal_moves(board, player):
            legal[r, c] = 1.0

        return np.stack([my, opp, legal], axis=0)   # (3, 8, 8)

    # ── Sélection d'action ─────────────────────────────────────────────────────

    def _greedy(self, obs: np.ndarray, legal_ids: List[int]) -> int:
        """Exploitation pure — argmax Q sur les actions légales uniquement."""
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online_net(x).squeeze(0).cpu().numpy()   # (65,)
        mask = np.full(65, -np.inf, dtype=np.float32)
        mask[legal_ids] = q[legal_ids]
        return int(np.argmax(mask))

    def select_action(self, obs: np.ndarray, legal_ids: List[int]) -> int:
        """Epsilon-greedy — utilisé pendant l'entraînement."""
        if self.rng.random() < self.eps:
            return self.rng.choice(legal_ids)
        return self._greedy(obs, legal_ids)

    def select_move(self, board: Board, player: int) -> Optional[Move]:
        """
        Interface select_move(board, player) → Optional[Move].
        Compatible avec tournament.py et othello_pygame.py.
        Exploitation pure (eps ignoré — setter eps=0 pour le tournoi).
        """
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        legal_ids = [encode_action(m) for m in legal]
        obs = self.board_to_obs(board, player)
        a   = self._greedy(obs, legal_ids)
        return decode_action(a)

    # ── Replay buffer ──────────────────────────────────────────────────────────

    def push(
        self,
        obs:        np.ndarray,
        action:     int,
        reward:     float,
        next_obs:   np.ndarray,
        done:       bool,
        next_legal: List[int],
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, done, next_legal)

    # ── Mise à jour gradient ───────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Un pas de gradient si le buffer est suffisamment rempli.
        Retourne la loss (float) ou None si l'entraînement ne démarre pas encore.

        Calcul de la cible (DQN standard avec masking légaux) :
          target = r + γ · max_{a ∈ légaux(s')} Q_target(s', a)   si non terminal
          target = r                                                si terminal
        """
        if len(self.buffer) < self.min_buffer_size:
            return None

        obs, actions, rewards, next_obs, dones, next_legal = \
            self.buffer.sample(self.batch_size, self.device)

        # Q(s, a) via le réseau online
        q_vals = self.online_net(obs)                                  # (B, 65)
        q_sa   = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)    # (B,)

        # max_a Q_target(s', a) avec masking des actions illégales
        with torch.no_grad():
            q_next = self.target_net(next_obs)             # (B, 65)
            q_next[~next_legal] = -float("inf")            # masque illégaux
            # Si TOUTES les actions sont masquées (état terminal sans coups),
            # max retourne -inf → remplacer par 0 pour éviter NaN dans la loss.
            max_next = q_next.max(dim=1).values            # (B,)
            max_next = torch.where(
                torch.isinf(max_next), torch.zeros_like(max_next), max_next
            )
            targets = rewards + self.gamma * max_next * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 0.5)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ── Décroissance epsilon ───────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ── Persistance ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "eps":        self.eps,
            "steps":      self.steps,
        }, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ck["online_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.eps   = ck["eps"]
        self.steps = ck["steps"]
        self.target_net.eval()
