# src/rl_v2/qlearning_v2.py
"""
Double Q-Learning agent (v2) — Q-table numpy.

Amélioration vs qlearning.py v1 :
  - Deux tables Q1 et Q2 indépendantes (Double Q-Learning, Hasselt 2010).
  - Réduit le biais optimiste du Q-Learning classique : Q1 sélectionne le
    meilleur coup en s', Q2 l'évalue (ou l'inverse, tiré à pile-ou-face).
  - La sélection d'action pendant le jeu utilise la moyenne Q1+Q2 (plus stable).
  - Q-tables stockées en np.ndarray float32 (469 350 × 65) au lieu de dicts Python
    → accès O(1) par index entier, pas de hashing, meilleure localité mémoire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import random

import numpy as np

from src.rl.features import State, N_STATES, N_ACTIONS, state_to_idx, _a_idx

ActionId = int  # -1 (PASS) ou 0..63


@dataclass
class QLearningAgentV2:
    alpha: float = 0.2
    gamma: float = 0.95
    eps: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9993
    seed: int = 0

    # Q-tables : shape (N_STATES, N_ACTIONS), float32, initialisées à zéro
    Q1: np.ndarray = field(default_factory=lambda: np.zeros((N_STATES, N_ACTIONS), dtype=np.float32))
    Q2: np.ndarray = field(default_factory=lambda: np.zeros((N_STATES, N_ACTIONS), dtype=np.float32))

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ── Accesseurs ─────────────────────────────────────────────────────────────

    def get_q1(self, s: State, a: ActionId) -> float:
        return float(self.Q1[state_to_idx(s), _a_idx(a)])

    def get_q2(self, s: State, a: ActionId) -> float:
        return float(self.Q2[state_to_idx(s), _a_idx(a)])

    def get_q(self, s: State, a: ActionId) -> float:
        """Moyenne des deux tables — estimation moins bruitée pour jouer."""
        si = state_to_idx(s)
        ai = _a_idx(a)
        return float(self.Q1[si, ai] + self.Q2[si, ai]) * 0.5

    # ── Sélection d'action ─────────────────────────────────────────────────────

    def best_action(self, s: State, legal_aids: List[ActionId]) -> ActionId:
        si = state_to_idx(s)
        best_a = legal_aids[0]
        best_q = float(self.Q1[si, _a_idx(best_a)] + self.Q2[si, _a_idx(best_a)])
        for a in legal_aids[1:]:
            q = float(self.Q1[si, _a_idx(a)] + self.Q2[si, _a_idx(a)])
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def select_action(self, s: State, legal_aids: List[ActionId]) -> ActionId:
        """Epsilon-greedy sur la moyenne Q1 + Q2."""
        if self.rng.random() < self.eps:
            return self.rng.choice(legal_aids)
        return self.best_action(s, legal_aids)

    # ── Mise à jour Double Q-Learning ─────────────────────────────────────────

    def update(
        self,
        s: State,
        a: ActionId,
        r: float,
        s2: State,
        legal_aids_s2: List[ActionId],
        done: bool,
    ) -> None:
        """
        Double Q-Learning (Hasselt, 2010).
        Avec 50 % de probabilité on met à jour Q1 (online=Q1, target=Q2),
        sinon Q2 (online=Q2, target=Q1).
        """
        si  = state_to_idx(s)
        ai  = _a_idx(a)
        si2 = state_to_idx(s2)

        if self.rng.random() < 0.5:
            q_online, q_target = self.Q1, self.Q2
        else:
            q_online, q_target = self.Q2, self.Q1

        old = float(q_online[si, ai])

        if done:
            target = r
        else:
            # Q_online choisit le meilleur coup dans s2...
            best_a2 = legal_aids_s2[0]
            best_v  = float(q_online[si2, _a_idx(best_a2)])
            for ap in legal_aids_s2[1:]:
                v = float(q_online[si2, _a_idx(ap)])
                if v > best_v:
                    best_v  = v
                    best_a2 = ap
            # ...Q_target en évalue la valeur
            target = r + self.gamma * float(q_target[si2, _a_idx(best_a2)])

        q_online[si, ai] = old + self.alpha * (target - old)

    # ── Décroissance epsilon ───────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # ── Nombre d'états visités (non-nuls) ─────────────────────────────────────

    def n_visited(self) -> int:
        """Nombre de lignes d'état ayant au moins une valeur non nulle dans Q1 ou Q2."""
        return int(np.any(self.Q1 != 0, axis=1).sum())
