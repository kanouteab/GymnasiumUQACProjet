# src/ui/othello_pygame.py
import time
import pygame

from src.envs.othello_env import (
    initial_board,
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
)

from src.agents.mcts import MCTSAgent
from src.agents.alphabeta import AlphaBetaAgent


# Bitboard helpers 
# Assumption: bit index = r*8 + c (row-major), r=0..7, c=0..7
def bit_at(bb: int, r: int, c: int) -> int:
    return (bb >> (r * 8 + c)) & 1


def count_bits(bb: int) -> int:
    return int(bb).bit_count()


# GUI 
CELL = 70
MARGIN = 30
TOPBAR = 70
BOARD_PX = CELL * 8
W = MARGIN * 2 + BOARD_PX
H = TOPBAR + MARGIN + BOARD_PX + MARGIN

FPS = 60

# Colors
BG = (18, 18, 20)
BOARD = (24, 120, 80)
GRID = (10, 60, 40)
HINT = (255, 215, 0)      # legal moves
BLACK = (20, 20, 20)
WHITE = (235, 235, 235)
TEXT = (240, 240, 240)
SUB = (180, 180, 180)
WIN = (120, 220, 140)
LOSE = (220, 120, 120)

def rc_from_mouse(pos):
    x, y = pos
    x -= MARGIN
    y -= TOPBAR
    if x < 0 or y < 0 or x >= BOARD_PX or y >= BOARD_PX:
        return None
    c = x // CELL
    r = y // CELL
    return int(r), int(c)

def draw_board(screen):
    # board background
    pygame.draw.rect(screen, BOARD, (MARGIN, TOPBAR, BOARD_PX, BOARD_PX), border_radius=10)
    # grid
    for i in range(9):
        # vertical
        pygame.draw.line(
            screen, GRID,
            (MARGIN + i * CELL, TOPBAR),
            (MARGIN + i * CELL, TOPBAR + BOARD_PX),
            2
        )
        # horizontal
        pygame.draw.line(
            screen, GRID,
            (MARGIN, TOPBAR + i * CELL),
            (MARGIN + BOARD_PX, TOPBAR + i * CELL),
            2
        )

def draw_pieces(screen, board):
    black_bb, white_bb = board
    for r in range(8):
        for c in range(8):
            cx = MARGIN + c * CELL + CELL // 2
            cy = TOPBAR + r * CELL + CELL // 2
            if bit_at(black_bb, r, c):
                pygame.draw.circle(screen, BLACK, (cx, cy), CELL // 2 - 6)
                pygame.draw.circle(screen, (60, 60, 60), (cx, cy), CELL // 2 - 6, 2)
            elif bit_at(white_bb, r, c):
                pygame.draw.circle(screen, WHITE, (cx, cy), CELL // 2 - 6)
                pygame.draw.circle(screen, (160, 160, 160), (cx, cy), CELL // 2 - 6, 2)

def draw_legal_moves(screen, legal_moves):
    for (r, c) in legal_moves:
        cx = MARGIN + c * CELL + CELL // 2
        cy = TOPBAR + r * CELL + CELL // 2
        pygame.draw.circle(screen, HINT, (cx, cy), 7)

def draw_text(screen, font, small, board, player, mode_text, speed_ms):
    black_bb, white_bb = board
    nb = count_bits(black_bb)
    nw = count_bits(white_bb)
    who = "Noir (●)" if player == 1 else "Blanc (○)"
    s = nb - nw

    title = font.render("Othello — GymnasiumUQACProjet", True, TEXT)
    screen.blit(title, (MARGIN, 15))

    line1 = small.render(f"À jouer : {who}   |   Score Noir={nb} Blanc={nw} (diff={s})", True, SUB)
    screen.blit(line1, (MARGIN, 42))

    line2 = small.render(f"Mode: {mode_text}   |   Vitesse: {speed_ms} ms/coup   |   [Espace] pause/lecture  [R] reset  [+/-] vitesse", True, SUB)
    screen.blit(line2, (MARGIN, H - MARGIN + 5))

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Othello UI")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Segoe UI", 22, bold=True)
    small = pygame.font.SysFont("Segoe UI", 16)

    # Agents (tu peux changer ici)
    agent_black = MCTSAgent(n_simulations=120, c_uct=1.4, rollout_max_steps=60, seed=0)
    agent_white = AlphaBetaAgent(depth=3, use_move_ordering=True)

    def reset_game():
        b = initial_board()
        p = 1
        if hasattr(agent_black, "reset_tree"): agent_black.reset_tree()
        if hasattr(agent_white, "reset_tree"): agent_white.reset_tree()
        return b, p

    board, player = reset_game()

    running = True
    autoplay = True          # True = ça joue automatiquement
    speed_ms = 200           # délai entre coups

    last_step_time = 0

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    autoplay = not autoplay
                elif event.key == pygame.K_r:
                    board, player = reset_game()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed_ms = max(0, speed_ms - 50)
                elif event.key == pygame.K_MINUS:
                    speed_ms = min(2000, speed_ms + 50)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # (Optionnel) clic humain si on met autoplay=False et qu'un joueur est humain
                pass

        screen.fill(BG)

        draw_board(screen)
        legal = get_legal_moves(board, player)
        draw_legal_moves(screen, legal)
        draw_pieces(screen, board)

        mode_text = "Auto (IA vs IA)" if autoplay else "Pause (Espace pour reprendre)"
        draw_text(screen, font, small, board, player, mode_text, speed_ms)

        # Step game
        if not is_terminal(board):
            now = pygame.time.get_ticks()
            if autoplay and (now - last_step_time) >= speed_ms:
                last_step_time = now

                move = agent_black.select_move(board, player) if player == 1 else agent_white.select_move(board, player)

                # sécurité
                if move is not None and legal and move not in legal:
                    move = legal[0]

                if move is not None:
                    board = apply_move(board, player, move)
                # PASS si None

                next_player = -player

                # Re-root (si MCTS)
                if hasattr(agent_black, "observe_move"):
                    agent_black.observe_move(move, board, next_player)
                if hasattr(agent_white, "observe_move"):
                    agent_white.observe_move(move, board, next_player)

                player = next_player
        else:
            wnr = get_winner(board)
            msg = "Nul" if wnr == 0 else ("Noir gagne" if wnr == 1 else "Blanc gagne")
            color = SUB if wnr == 0 else (WIN if wnr == 1 else LOSE)
            banner = font.render(msg, True, color)
            screen.blit(banner, (W - MARGIN - banner.get_width(), 15))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
