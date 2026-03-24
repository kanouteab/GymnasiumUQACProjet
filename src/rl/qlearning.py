# src/rl_v2/qlearning_v2.py
"""
Double Q-Learning agent (v2).

Amélioration vs qlearning.py v1 :
  - Deux tables Q1 et Q2 indépendantes (Double Q-Learning, Hasselt 2010).
  - Réduit le biais optimiste du Q-Learning classique : Q1 sélectionne le
    meilleur coup en s', Q2 l'évalue (ou l'inverse, tiré à pile-ou-face).
  - La sélection d'action pendant le jeu utilise la moyenne Q1+Q2 (plus stable).
  - eps_decay par défaut adapté à 5 000 épisodes (0.9993 au lieu de 0.995).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import random

from Atelier2.GymnasiumUQACProjet.src.rl.features import State

ActionId = int  # -1 (PASS) ou 0..63


@dataclass
class QLearningAgentV2:
    alpha: float = 0.2
    gamma: float = 0.95
    eps: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9993   # ~1 000 épisodes pour atteindre eps_min sur 5 000 épisodes
    seed: int = 0

    Q1: Dict[Tuple[State, ActionId], float] = field(default_factory=dict)
    Q2: Dict[Tuple[State, ActionId], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    # ── Accesseurs ─────────────────────────────────────────────────────────────

    def get_q1(self, s: State, a: ActionId) -> float:
        return self.Q1.get((s, a), 0.0)

    def get_q2(self, s: State, a: ActionId) -> float:
        return self.Q2.get((s, a), 0.0)

    def get_q(self, s: State, a: ActionId) -> float:
        """Moyenne des deux tables — estimation moins bruitée pour jouer."""
        return (self.get_q1(s, a) + self.get_q2(s, a)) / 2.0

    # ── Sélection d'action ─────────────────────────────────────────────────────

    def best_action(self, s: State, legal_aids: List[ActionId]) -> ActionId:
        best_a = legal_aids[0]
        best_q = self.get_q(s, best_a)
        for a in legal_aids[1:]:
            q = self.get_q(s, a)
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
        Double Q-Learning (Hasselt, 2010) :

        Avec 50 % de probabilité :
          - Q_online = Q1, Q_target = Q2  →  met à jour Q1
        Sinon :
          - Q_online = Q2, Q_target = Q1  →  met à jour Q2

        L'action de bootstrap est choisie par Q_online (argmax),
        mais évaluée par Q_target (valeur). Cela supprime le biais
        optimiste du Q-Learning standard où la même table sélectionne
        ET évalue.
        """
        if self.rng.random() < 0.5:
            q_online, q_target = self.Q1, self.Q2
        else:
            q_online, q_target = self.Q2, self.Q1

        old = q_online.get((s, a), 0.0)

        if done:
            target = r
        else:
            # Q_online choisit le meilleur coup dans s2...
            best_a = max(legal_aids_s2, key=lambda ap: q_online.get((s2, ap), 0.0))
            # ...Q_target en évalue la valeur
            target = r + self.gamma * q_target.get((s2, best_a), 0.0)

        q_online[(s, a)] = old + self.alpha * (target - old)

    # ── Décroissance epsilon ───────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
