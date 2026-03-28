# src/experiments/plot_results.py
"""
Génération des graphiques pour le rapport.

  plot_learning_curve      : courbe win-rate + ε (Q-Learning, 2 panneaux)
  plot_dqn_learning_curve  : courbe win-rate + loss + ε (DQN, 3 panneaux)
  plot_tournament          : heatmap NxN des win rates inter-agents
  plot_final_eval          : barres horizontales des win rates finaux
                             (paramètre prefix pour distinguer QL / DQN)
"""
from __future__ import annotations

import csv
import os
from typing import List


# ── Courbe d'apprentissage ─────────────────────────────────────────────────────

def plot_learning_curve(
    csv_path: str = "artifacts/training_stats.csv",
    out_dir:  str = "artifacts",
) -> str:
    """
    Lit artifacts/training_stats.csv (produit par train_rl.main()) et génère
    un graphique à deux panneaux :
      - haut  : taux de victoire (fenêtre glissante log_every épisodes)
      - bas   : valeur de epsilon (exploration)
    Les bandes de fond indiquent la phase du curriculum.
    Sauvegarde dans out_dir/learning_curve.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ── Lecture ────────────────────────────────────────────────────────────────
    ep_list:    List[int]   = []
    wr_list:    List[float] = []
    eps_list:   List[float] = []
    phase_list: List[str]   = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep_list.append(int(row["ep"]))
            wr_list.append(float(row["win_rate"]))
            eps_list.append(float(row["eps"]))
            phase_list.append(row["phase"])

    if not ep_list:
        print("Aucune donnée dans le CSV — courbe non générée.")
        return ""

    log_every = ep_list[1] - ep_list[0] if len(ep_list) > 1 else 200

    # ── Couleurs des phases ────────────────────────────────────────────────────
    PHASE_COLORS = {
        "Random":       "#aed6f1",
        "MCTS":         "#a9dfbf",
        "MCTS-200":     "#a9dfbf",
        "AlphaBeta-d2": "#f9e79f",
        "AlphaBeta-d3": "#f0b27a",
        "AlphaBeta-d4": "#d7bde2",
    }

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle("Courbe d'apprentissage — Q-Learning", fontsize=13, fontweight="bold")

    # Bandes de phase (regroupement des lignes consécutives de même phase)
    legend_phases: List[str] = []
    i = 0
    while i < len(phase_list):
        phase = phase_list[i]
        j = i
        while j < len(phase_list) and phase_list[j] == phase:
            j += 1
        # La bande commence à l'épisode 0 de la phase (juste avant le 1er log)
        x0 = ep_list[i] - log_every
        x1 = ep_list[j - 1]
        color = PHASE_COLORS.get(phase, "#eeeeee")
        for ax in (ax1, ax2):
            ax.axvspan(x0, x1, alpha=0.25, color=color, zorder=0)
        if phase not in legend_phases:
            legend_phases.append(phase)
        i = j

    # Panneau 1 : win rate
    ax1.plot(ep_list, wr_list, color="#1a5276", linewidth=1.6, zorder=2)
    ax1.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=0.9, alpha=0.7, label="50 %")
    ax1.set_ylabel("Taux de victoire\n(fenêtre glissante)", fontsize=10)
    ax1.set_ylim(0.0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)

    patches = [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.7, label=f"Phase : {p}")
        for p in legend_phases if p in PHASE_COLORS
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)

    # Panneau 2 : epsilon
    ax2.plot(ep_list, eps_list, color="#c0392b", linewidth=1.4, zorder=2)
    ax2.set_ylabel("ε", fontsize=10)
    ax2.set_xlabel("Épisode", fontsize=10)
    ax2.set_ylim(0.0, 1.08)
    ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbe d'apprentissage → {out_path}")
    return out_path


# ── Courbe d'apprentissage DQN ─────────────────────────────────────────────────

def plot_dqn_learning_curve(
    csv_path: str = "artifacts/dqn_training_stats.csv",
    out_dir:  str = "artifacts",
) -> str:
    """
    Lit artifacts/dqn_training_stats.csv (produit par train_dqn.main()) et génère
    un graphique à trois panneaux :
      - haut   : taux de victoire (fenêtre glissante log_every épisodes)
      - milieu : perte moyenne (Huber loss)
      - bas    : valeur de epsilon
    Les bandes de fond indiquent la phase du curriculum.
    Sauvegarde dans out_dir/dqn_learning_curve.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ep_list:    List[int]   = []
    wr_list:    List[float] = []
    eps_list:   List[float] = []
    loss_list:  List[float] = []
    phase_list: List[str]   = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep_list.append(int(row["ep"]))
            wr_list.append(float(row["win_rate"]))
            eps_list.append(float(row["eps"]))
            loss_list.append(float(row.get("avg_loss", 0.0)))
            phase_list.append(row["phase"])

    if not ep_list:
        print("Aucune donnée dans le CSV DQN — courbe non générée.")
        return ""

    log_every = ep_list[1] - ep_list[0] if len(ep_list) > 1 else 200

    PHASE_COLORS = {
        "Random":       "#aed6f1",
        "MCTS":         "#a9dfbf",
        "MCTS-200":     "#a9dfbf",
        "AlphaBeta-d2": "#f9e79f",
        "AlphaBeta-d3": "#f0b27a",
        "AlphaBeta-d4": "#d7bde2",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1]},
    )
    fig.suptitle("Courbe d'apprentissage — DQN (CNN 3 canaux)", fontsize=13, fontweight="bold")

    # Bandes de phase
    legend_phases: List[str] = []
    i = 0
    while i < len(phase_list):
        phase = phase_list[i]
        j = i
        while j < len(phase_list) and phase_list[j] == phase:
            j += 1
        x0 = ep_list[i] - log_every
        x1 = ep_list[j - 1]
        color = PHASE_COLORS.get(phase, "#eeeeee")
        for ax in (ax1, ax2, ax3):
            ax.axvspan(x0, x1, alpha=0.25, color=color, zorder=0)
        if phase not in legend_phases:
            legend_phases.append(phase)
        i = j

    # Panneau 1 : win rate
    ax1.plot(ep_list, wr_list, color="#1a5276", linewidth=1.6, zorder=2)
    ax1.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_ylabel("Taux de victoire\n(fenêtre glissante)", fontsize=10)
    ax1.set_ylim(0.0, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)
    patches = [
        mpatches.Patch(color=PHASE_COLORS[p], alpha=0.7, label=f"Phase : {p}")
        for p in legend_phases if p in PHASE_COLORS
    ]
    ax1.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)

    # Panneau 2 : loss
    ax2.plot(ep_list, loss_list, color="#884ea0", linewidth=1.4, zorder=2)
    ax2.set_ylabel("Perte moyenne\n(Huber)", fontsize=10)
    ax2.set_ylim(bottom=0.0)
    ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)

    # Panneau 3 : epsilon
    ax3.plot(ep_list, eps_list, color="#c0392b", linewidth=1.4, zorder=2)
    ax3.set_ylabel("ε", fontsize=10)
    ax3.set_xlabel("Épisode", fontsize=10)
    ax3.set_ylim(0.0, 1.08)
    ax3.grid(axis="y", linestyle=":", alpha=0.5, zorder=1)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dqn_learning_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Courbe d'apprentissage DQN → {out_path}")
    return out_path


# ── Heatmap tournoi ────────────────────────────────────────────────────────────

def plot_tournament(
    matrix:      List[List[float]],
    agent_names: List[str],
    out_dir:     str = "artifacts",
    out_path:    str = "",
) -> str:
    """
    Heatmap NxN des résultats du tournoi.
    matrix[i][j] = taux de victoire de l'agent i (Noirs) contre l'agent j (Blancs).
    Une colonne « Moy. » est ajoutée à droite avec la moyenne sur la ligne.
    Sauvegarde dans out_dir/tournament.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n   = len(agent_names)
    mat = np.array(matrix, dtype=float)

    # Moy. Noirs : win rate moyen en jouant Noirs (sans diagonale)
    mask_diag = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask_diag, False)
    mat_off   = np.where(mask_diag, mat, np.nan)
    avg_black = np.nanmean(mat_off, axis=1, keepdims=True)

    # Moy. Blancs : win rate moyen en jouant Blancs
    # agent i joue Blancs contre j  →  i gagne si j (Noirs) perd  →  1 - mat[j,i]
    mat_as_white = 1.0 - mat.T
    np.fill_diagonal(mat_as_white, np.nan)
    avg_white = np.nanmean(mat_as_white, axis=1, keepdims=True)

    # Moy. globale : moyenne des deux
    avg_global = (avg_black + avg_white) / 2.0

    mat_display = np.hstack([mat, avg_black, avg_global])
    col_labels  = agent_names + ["Moy. N", "Moy. N+B"]

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(mat_display, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(agent_names, fontsize=9)
    ax.set_xlabel("Agent Blanc (adversaire)", fontsize=10)
    ax.set_ylabel("Agent Noir", fontsize=10)
    ax.set_title(
        "Tournoi inter-agents\n(taux de victoire — Moy. N+B = moyenne toutes couleurs)",
        fontsize=12, fontweight="bold",
    )

    # Valeurs dans les cellules
    n_extra = 2  # nombre de colonnes supplémentaires
    for i in range(n):
        for j in range(len(col_labels)):
            val = mat_display[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if (val < 0.30 or val > 0.72) else "black"
            suffix    = " (—)" if (j < n and i == j) else ""
            is_extra  = j >= n
            ax.text(
                j, i, f"{val:.0%}{suffix}",
                ha="center", va="center", fontsize=8,
                color=text_color,
                fontweight="bold"   if is_extra else "normal",
                fontstyle="italic"  if is_extra else "normal",
            )

    # Lignes verticales séparant les colonnes synthèse
    ax.axvline(n - 0.5,         color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axvline(n + 1 - 0.5,    color="black", linewidth=1.0, linestyle=":",  alpha=0.5)

    plt.colorbar(im, ax=ax, label="Taux de victoire", shrink=0.85)
    plt.tight_layout()
    if not out_path:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "tournament.png")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap du tournoi → {out_path}")
    return out_path


# ── Barres de l'évaluation finale ─────────────────────────────────────────────

def plot_final_eval(
    results: dict,
    out_dir: str = "artifacts",
    prefix:  str = "",
) -> str:
    """
    Graphique en barres horizontales des win rates finaux de l'agent RL
    contre chaque adversaire.
    results = {"vs Random": 0.78, "vs MCTS": 0.45, ...}
    Sauvegarde dans out_dir/final_eval.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    values = list(results.values())
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0.5, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Taux de victoire (agent RL en Noirs)", fontsize=10)
    title = f"Évaluation finale — {prefix.rstrip('_').upper() if prefix else 'QL'} vs chaque adversaire"
    ax.set_title(title, fontsize=11, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=9,
        )

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}final_eval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Évaluation finale → {out_path}")
    return out_path
