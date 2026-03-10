# src/rl/qlearning.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import random

State = Tuple[int, int, int, int]  # from features.py
ActionId = int  # -1 or 0..63

@dataclass
class QLearningAgent:
    alpha: float = 0.2
    gamma: float = 0.95
    eps: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.995
    seed: int = 0
    Q: Dict[Tuple[State, ActionId], float] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def get_q(self, s: State, a: ActionId) -> float:
        return self.Q.get((s, a), 0.0)

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
        # epsilon-greedy
        if self.rng.random() < self.eps:
            return self.rng.choice(legal_aids)
        return self.best_action(s, legal_aids)

    def update(self, s: State, a: ActionId, r: float, s2: State, legal_aids_s2: List[ActionId], done: bool):
        old = self.get_q(s, a)
        if done:
            target = r
        else:
            # max_a' Q(s',a')
            m = self.get_q(s2, legal_aids_s2[0])
            for ap in legal_aids_s2[1:]:
                q = self.get_q(s2, ap)
                if q > m:
                    m = q
            target = r + self.gamma * m

        self.Q[(s, a)] = old + self.alpha * (target - old)

    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)