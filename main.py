import chess
import ChessEngine
import random
import numpy as np

def get_legal_moves(generator_):
    l = []
    for i in generator_:
        l.append(i)
    return l

def choose_best_move(legal_moves, board, depth=2):
    print("Thinking...")
    best_move = ('', float('inf'))
    start_fen = board.fen()
    frontier = legal_moves
    for move in frontier:
        my_board = chess.Board(start_fen)
        my_board.push(move)
        print(my_board)
        engine = ChessEngine.Engine(my_board)
        fen = my_board.fen()
        # print(fen)
        eval = engine.run_engine([fen],'Model_saves/100KChess_64.joblib')
        if eval < best_move[1]:
            best_move = (move,eval)
    print('BEST MOVE:',best_move)
    return best_move[0]

# ----------------------------

board = chess.Board()
draw = board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves()

while True:
    players_move = 1
    while players_move:
        try:
            move = input("\n\nYour move: ")
            board.push_san(move)
            players_move = 0
        except Exception:
            print("Move is invalid")

    if board.is_checkmate():
       print('Game over!\nYou Won!')
       break
    if draw:
        print("It is a draw")
        break

    legal_moves = get_legal_moves(board.legal_moves)    
    ai_move = choose_best_move(legal_moves, board)
    
    print(f'I will play: {ai_move}\n')
    board.push(ai_move)
    print(board)

    if board.is_checkmate():
       print('Game over!\nI Won!')
       break
    if draw:
        print("It is a draw")
        break