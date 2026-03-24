"""
Command to launch evaluation:
python -m src.experiments.eval_rl_vs_mcts
"""
import pickle
from src.rl.qlearning import QLearningAgent
from src.agents.mcts import MCTSAgent
from src.envs.othello_env import (
    initial_board,
    get_legal_moves,
    apply_move,
    is_terminal,
    get_winner,
    score,
)
from src.rl.features import state_features, action_to_id, id_to_action


def play_game(agent_black, agent_white):
    board = initial_board()
    player = 1  # noir commence

    # reset arbre MCTS si nécessaire
    if hasattr(agent_black, "reset_tree"):
        agent_black.reset_tree()
    if hasattr(agent_white, "reset_tree"):
        agent_white.reset_tree()

    while not is_terminal(board):
        legal = get_legal_moves(board, player)

        if player == 1:
            # RL joue en noir
            if not legal:
                move = None
            else:
                s = state_features(board, 1)
                legal_aids = [action_to_id(m) for m in legal]
                best_a = agent_black.best_action(s, legal_aids)
                move = id_to_action(best_a)
        else:
            # MCTS joue en blanc
            move = agent_white.select_move(board, player)

        # sécurité : si coup invalide, on passe au premier coup légal
        if move is not None:
            if move not in legal:
                move = legal[0] if legal else None
            if move is not None:
                board = apply_move(board, player, move)

        next_player = -player

        # informer les agents si besoin (utile pour MCTS)
        if hasattr(agent_black, "observe_move"):
            agent_black.observe_move(move, board, next_player)
        if hasattr(agent_white, "observe_move"):
            agent_white.observe_move(move, board, next_player)

        player = next_player

    return get_winner(board), score(board)

if __name__ == "__main__":
    # Charger la Q-table entraînée
    agent_rl = QLearningAgent(eps=0.0)
    with open("artifacts/qtable.pkl", "rb") as f:
        agent_rl.Q = pickle.load(f)

    # Agent MCTS
    agent_mcts = MCTSAgent(n_simulations=200)

    wins = {1: 0, -1: 0, 0: 0}
    games = 50

    for _ in range(games):
        winner, _ = play_game(agent_rl, agent_mcts)
        wins[winner] += 1

    print("RL (Black) vs MCTS (White)")
    print("Games:", games)
    print("RL wins:", wins[1])
    print("MCTS wins:", wins[-1])

    print("Draw:", wins[0])
