# src/ui/othello_pygame.py
import os
import pickle
import random

import pygame

from src.envs.othello_env import (
    initial_board,
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
)
from src.agents.mcts import MCTSAgent
from src.agents.alphabeta import AlphaBetaAgent


# ── Bitboard helpers ───────────────────────────────────────────────────────────

def bit_at(bb: int, r: int, c: int) -> int:
    return (bb >> (r * 8 + c)) & 1

def count_bits(bb: int) -> int:
    return int(bb).bit_count()


# ── Agent wrappers ─────────────────────────────────────────────────────────────

class _RandomAgent:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def select_move(self, board, player):
        legal = get_legal_moves(board, player)
        return self.rng.choice(legal) if legal else None


class _QLv1Agent:
    """Wraps QLearningAgent v1 avec select_move()."""
    def __init__(self, qtable_path="artifacts/qtable.pkl"):
        from src.rl.qlearning import QLearningAgent
        from src.rl.features import state_features, id_to_action, action_to_id
        self._sf = state_features
        self._ia = id_to_action
        self._ai = action_to_id
        self.agent = QLearningAgent(eps=0.0)
        if os.path.exists(qtable_path):
            with open(qtable_path, "rb") as f:
                self.agent.Q = pickle.load(f)

    def select_move(self, board, player):
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        s = self._sf(board, player)
        aids = [self._ai(mv) for mv in legal]
        a = self.agent.best_action(s, aids)
        return self._ia(a)


class _QLv2Agent:
    """Wraps QLearningAgentV2 avec select_move()."""
    def __init__(self, qtable_path="artifacts/qtable_v2.pkl"):
        from Atelier2.GymnasiumUQACProjet.src.rl.qlearning import QLearningAgentV2
        from Atelier2.GymnasiumUQACProjet.src.rl.features import state_features_v2, id_to_action, action_to_id
        self._sf = state_features_v2
        self._ia = id_to_action
        self._ai = action_to_id
        self.agent = QLearningAgentV2(eps=0.0)
        if os.path.exists(qtable_path):
            with open(qtable_path, "rb") as f:
                data = pickle.load(f)
                self.agent.Q1 = data["Q1"]
                self.agent.Q2 = data["Q2"]

    def select_move(self, board, player):
        legal = get_legal_moves(board, player)
        if not legal:
            return None
        s = self._sf(board, player)
        aids = [self._ai(mv) for mv in legal]
        a = self.agent.best_action(s, aids)
        return self._ia(a)


# ── Agent catalogue ────────────────────────────────────────────────────────────

AGENT_SPECS = [
    {"label": "Humain",           "kind": "human"},
    {"label": "Random",           "kind": "random"},
    {"label": "Alpha-Beta d=2",   "kind": "ab",   "depth": 2},
    {"label": "Alpha-Beta d=4",   "kind": "ab",   "depth": 4},
    {"label": "MCTS (120 sims)",  "kind": "mcts", "sims": 120},
    {"label": "Q-Learning v1",    "kind": "ql1",  "path": "artifacts/qtable.pkl"},
    {"label": "Q-Learning v2",    "kind": "ql2",  "path": "artifacts/qtable_v2.pkl"},
]


def make_agent(spec):
    kind = spec["kind"]
    if kind == "human":   return None
    if kind == "random":  return _RandomAgent()
    if kind == "ab":      return AlphaBetaAgent(depth=spec["depth"], use_move_ordering=True)
    if kind == "mcts":    return MCTSAgent(n_simulations=spec["sims"], c_uct=1.4,
                                           rollout_max_steps=60, seed=0)
    if kind == "ql1":     return _QLv1Agent(spec["path"])
    if kind == "ql2":     return _QLv2Agent(spec["path"])
    return None


# ── Layout & Colors ────────────────────────────────────────────────────────────

CELL     = 70
MARGIN   = 30
TOPBAR   = 70
BOARD_PX = CELL * 8
W        = MARGIN * 2 + BOARD_PX          # 620
H        = TOPBAR + MARGIN + BOARD_PX + MARGIN   # 690

FPS = 60

BG    = (18,  18,  20)
BOARD = (24,  120, 80)
GRID  = (10,  60,  40)
HINT  = (255, 215, 0)
BLACK = (20,  20,  20)
WHITE = (235, 235, 235)
TEXT  = (240, 240, 240)
SUB   = (160, 160, 160)
WIN   = (120, 220, 140)
LOSE  = (220, 120, 120)
ACC   = (60,  160, 100)
SEL   = (35,  90,  55)
HOVER = (45,  45,  58)


# ── Board drawing helpers ──────────────────────────────────────────────────────

def rc_from_mouse(pos):
    x, y = pos
    x -= MARGIN
    y -= TOPBAR
    if x < 0 or y < 0 or x >= BOARD_PX or y >= BOARD_PX:
        return None
    return int(y // CELL), int(x // CELL)


def draw_board(screen):
    pygame.draw.rect(screen, BOARD, (MARGIN, TOPBAR, BOARD_PX, BOARD_PX), border_radius=10)
    for i in range(9):
        pygame.draw.line(screen, GRID,
                         (MARGIN + i*CELL, TOPBAR), (MARGIN + i*CELL, TOPBAR + BOARD_PX), 2)
        pygame.draw.line(screen, GRID,
                         (MARGIN, TOPBAR + i*CELL), (MARGIN + BOARD_PX, TOPBAR + i*CELL), 2)


def draw_pieces(screen, board):
    black_bb, white_bb = board
    for r in range(8):
        for c in range(8):
            cx = MARGIN + c*CELL + CELL//2
            cy = TOPBAR + r*CELL + CELL//2
            if bit_at(black_bb, r, c):
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL//2 - 6)
                pygame.draw.circle(screen, (60, 60, 60), (cx, cy), CELL//2 - 6, 2)
            elif bit_at(white_bb, r, c):
                pygame.draw.circle(screen, WHITE, (cx, cy), CELL//2 - 6)
                pygame.draw.circle(screen, (160, 160, 160), (cx, cy), CELL//2 - 6, 2)


def draw_legal_hints(screen, legal_moves, human_turn):
    """Indicateurs de coups légaux — affichés uniquement pour le joueur humain."""
    if not human_turn:
        return
    for (r, c) in legal_moves:
        cx = MARGIN + c*CELL + CELL//2
        cy = TOPBAR + r*CELL + CELL//2
        pygame.draw.circle(screen, HINT, (cx, cy), 8)
        pygame.draw.circle(screen, (200, 170, 0), (cx, cy), 8, 2)


def draw_game_hud(screen, font, small, board, player, black_lbl, white_lbl, speed_ms):
    """HUD en jeu — retourne back_rect (bouton Retour menu)."""
    black_bb, white_bb = board
    nb = count_bits(black_bb)
    nw = count_bits(white_bb)
    who = "● Noirs" if player == 1 else "○ Blancs"

    # Titre + score
    title = font.render("Othello", True, TEXT)
    screen.blit(title, (MARGIN, 12))

    score_str = f"● {black_lbl}  {nb} — {nw}  {white_lbl} ○   |   À jouer : {who}"
    sc = small.render(score_str, True, SUB)
    screen.blit(sc, (MARGIN, 40))

    # Bouton retour menu
    back_rect = pygame.Rect(W - MARGIN - 95, 8, 85, 30)
    pygame.draw.rect(screen, (50, 50, 62), back_rect, border_radius=6)
    pygame.draw.rect(screen, (80, 80, 100), back_rect, 1, border_radius=6)
    back_lbl = small.render("◀  Menu", True, TEXT)
    screen.blit(back_lbl, (back_rect.x + 10, back_rect.y + 7))

    # Barre du bas
    hint = "  [Espace] pause  [R] reset  [+/-] vitesse IA  [Échap] menu"
    bottom = small.render(hint, True, (100, 100, 100))
    screen.blit(bottom, (MARGIN, H - MARGIN + 5))

    return back_rect


# ── Menu drawing ───────────────────────────────────────────────────────────────

def draw_menu(screen, font, title_font, small, selected, hover_item):
    """
    selected    : {1: idx, -1: idx}  — index dans AGENT_SPECS
    hover_item  : (player, idx) | "start" | None

    Retourne (buttons, start_rect)
      buttons : [(rect, player, idx), ...]
    """
    screen.fill(BG)

    # Titre
    title = title_font.render("Othello", True, TEXT)
    screen.blit(title, (W//2 - title.get_width()//2, 14))
    sub = small.render("Choisissez les joueurs, puis lancez la partie.", True, SUB)
    screen.blit(sub, (W//2 - sub.get_width()//2, 46))

    # Séparateur vertical
    pygame.draw.line(screen, (50, 50, 65), (W//2, 75), (W//2, H - 100), 1)

    ROW_H   = 46
    FIRST_Y = 100
    col_w   = W//2 - MARGIN - 20

    columns = [
        (1,  "● Noirs",  MARGIN + 8),
        (-1, "○ Blancs", W//2 + 12),
    ]

    buttons = []
    for player, header, col_x in columns:
        hdr = font.render(header, True, TEXT)
        screen.blit(hdr, (col_x, FIRST_Y - 28))

        for i, spec in enumerate(AGENT_SPECS):
            rect = pygame.Rect(col_x, FIRST_Y + i * ROW_H, col_w, ROW_H - 5)
            is_sel = selected[player] == i
            is_hov = hover_item == (player, i)

            if is_sel:
                bg, border = SEL, ACC
            elif is_hov:
                bg, border = HOVER, (75, 75, 95)
            else:
                bg, border = (28, 28, 36), (50, 50, 65)

            pygame.draw.rect(screen, bg, rect, border_radius=8)
            pygame.draw.rect(screen, border, rect, 2, border_radius=8)

            # Radio dot
            dx, dy = rect.x + 15, rect.centery
            pygame.draw.circle(screen, border, (dx, dy), 7, 2)
            if is_sel:
                pygame.draw.circle(screen, ACC, (dx, dy), 4)

            lbl_col = TEXT if is_sel else SUB
            lbl = font.render(spec["label"], True, lbl_col)
            screen.blit(lbl, (rect.x + 30, rect.centery - lbl.get_height()//2))

            # Badge "non entraîné" pour les QL sans fichier
            if spec["kind"] in ("ql1", "ql2") and not os.path.exists(spec["path"]):
                na = small.render("non entraîné", True, (190, 80, 80))
                screen.blit(na, (rect.right - na.get_width() - 8,
                                 rect.centery - na.get_height()//2))

            buttons.append((rect, player, i))

    # Bouton Lancer
    btn_y = FIRST_Y + len(AGENT_SPECS) * ROW_H + 18
    start_rect = pygame.Rect(W//2 - 135, btn_y, 270, 54)
    is_hov = hover_item == "start"
    btn_col = (65, 175, 110) if is_hov else (42, 128, 78)
    pygame.draw.rect(screen, btn_col, start_rect, border_radius=14)
    s_lbl = title_font.render("▶  Lancer la partie", True, TEXT)
    screen.blit(s_lbl, (start_rect.centerx - s_lbl.get_width()//2,
                         start_rect.centery - s_lbl.get_height()//2))

    return buttons, start_rect


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Othello")
    clock = pygame.time.Clock()

    font       = pygame.font.SysFont("Segoe UI", 18, bold=True)
    title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    small      = pygame.font.SysFont("Segoe UI", 14)

    # ── State ──────────────────────────────────────────────────────────────────
    STATE_MENU = "menu"
    STATE_GAME = "game"
    state = STATE_MENU

    # Menu
    selected   = {1: 0, -1: 2}   # défaut : Humain (Noir) vs Alpha-Beta d=2 (Blanc)
    hover_item = None
    menu_buttons = []
    menu_start   = pygame.Rect(0, 0, 0, 0)

    # Jeu
    board       = initial_board()
    player      = 1
    agents      = {1: None, -1: None}
    black_lbl   = ""
    white_lbl   = ""
    speed_ms    = 400
    autoplay    = True
    last_step   = 0
    human_move  = None   # coup posé par le joueur humain via clic
    back_rect   = pygame.Rect(0, 0, 0, 0)

    def start_game():
        nonlocal board, player, agents, black_lbl, white_lbl
        nonlocal last_step, human_move, autoplay
        spec_b = AGENT_SPECS[selected[1]]
        spec_w = AGENT_SPECS[selected[-1]]
        agents[1]  = make_agent(spec_b)
        agents[-1] = make_agent(spec_w)
        black_lbl  = spec_b["label"]
        white_lbl  = spec_w["label"]
        board      = initial_board()
        player     = 1
        human_move = None
        autoplay   = True
        last_step  = 0
        for ag in agents.values():
            if ag is not None and hasattr(ag, "reset_tree"):
                ag.reset_tree()

    # ── Game loop ──────────────────────────────────────────────────────────────
    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()

        # ── Events ─────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif state == STATE_MENU:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, pl, idx in menu_buttons:
                        if rect.collidepoint(event.pos):
                            selected[pl] = idx
                    if menu_start.collidepoint(event.pos):
                        start_game()
                        state = STATE_GAME

            elif state == STATE_GAME:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,):
                        state = STATE_MENU
                    elif event.key == pygame.K_r:
                        start_game()
                    elif event.key == pygame.K_SPACE:
                        autoplay = not autoplay
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        speed_ms = max(0, speed_ms - 50)
                    elif event.key == pygame.K_MINUS:
                        speed_ms = min(3000, speed_ms + 50)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if back_rect.collidepoint(event.pos):
                        state = STATE_MENU
                    # Coup humain
                    elif agents[player] is None and not is_terminal(board):
                        rc = rc_from_mouse(event.pos)
                        legal = get_legal_moves(board, player)
                        if rc and rc in legal:
                            human_move = rc

        # ── Draw ───────────────────────────────────────────────────────────────
        if state == STATE_MENU:
            # Calcul hover
            hover_item = None
            for rect, pl, idx in menu_buttons:
                if rect.collidepoint(mouse_pos):
                    hover_item = (pl, idx)
                    break
            if menu_start.collidepoint(mouse_pos):
                hover_item = "start"

            menu_buttons, menu_start = draw_menu(
                screen, font, title_font, small, selected, hover_item
            )

        elif state == STATE_GAME:
            screen.fill(BG)
            legal      = get_legal_moves(board, player)
            human_turn = agents[player] is None

            draw_board(screen)
            draw_legal_hints(screen, legal, human_turn)
            draw_pieces(screen, board)
            back_rect = draw_game_hud(
                screen, font, small, board, player,
                black_lbl, white_lbl, speed_ms
            )

            # Bannière de fin
            if is_terminal(board):
                wnr = get_winner(board)
                if wnr == 0:
                    msg, color = "Match nul", SUB
                elif wnr == 1:
                    msg, color = f"Noir gagne !  ({black_lbl})", WIN
                else:
                    msg, color = f"Blanc gagne !  ({white_lbl})", LOSE
                banner = title_font.render(msg, True, color)
                bx = W//2 - banner.get_width()//2
                overlay = pygame.Surface((banner.get_width() + 24, 36), pygame.SRCALPHA)
                overlay.fill((10, 10, 14, 210))
                screen.blit(overlay, (bx - 12, 8))
                screen.blit(banner, (bx, 10))

            # Tour humain — indice textuel
            elif human_turn:
                hint = small.render(
                    "Votre tour — cliquez sur une case surlignée", True, HINT
                )
                screen.blit(hint, (W//2 - hint.get_width()//2, H - MARGIN + 5))

                # Consommer le clic s'il y a lieu
                if human_move is not None:
                    move = human_move
                    human_move = None
                    board  = apply_move(board, player, move)
                    next_p = -player
                    for ag in agents.values():
                        if ag is not None and hasattr(ag, "observe_move"):
                            ag.observe_move(move, board, next_p)
                    player = next_p

            # Tour IA
            else:
                now = pygame.time.get_ticks()
                if autoplay and (now - last_step) >= speed_ms:
                    last_step = now
                    move = agents[player].select_move(board, player)
                    if move is not None and legal and move not in legal:
                        move = legal[0] if legal else None
                    if move is not None:
                        board = apply_move(board, player, move)
                    next_p = -player
                    for ag in agents.values():
                        if ag is not None and hasattr(ag, "observe_move"):
                            ag.observe_move(move, board, next_p)
                    player = next_p

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()

