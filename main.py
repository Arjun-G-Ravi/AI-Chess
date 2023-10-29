import chess
import ChessEngine
from ChessEngine import ChessNN
import numpy as np
from copy import deepcopy

def get_legal_moves(generator_):
    l = []
    for i in generator_:
        l.append(i)
    return l

def choose_best_move(legal_moves, board):
    print("Thinking...")
    best_move = ('', -float('inf'))
    start_fen = board.fen()
    frontier = legal_moves
    my_board = chess.Board(start_fen)
    new_frontier = []
    for move in frontier:
        current_board = deepcopy(my_board)
        current_board.push(move)
        leg_moves = get_legal_moves(current_board.legal_moves) 
        for m in leg_moves:
            new_frontier.append((move, m))

    tot = len(new_frontier)
    ct = 0
    for move in new_frontier:
        this_board = deepcopy(my_board)
        for m in move:
            this_board.push(m)
        # print(this_board) # This is 2 moves into the future
        
        engine = ChessEngine.Engine(this_board)
        fen = this_board.fen()
        eval = engine.run_engine([fen], model='/home/arjun/Desktop/GitHub/AI-Chess/Model_saves/Pytorch_v1.joblib')
        if eval > best_move[1]:
            best_move = (move,eval)
        ct += 1
        print(f'Progress: {ct}/ {tot}')
    print('BEST MOVE:',best_move)
    return best_move[0][0]

# --- main --- 

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
            if move == 'ls':
                print('Possible moves are: \n')
                legals = get_legal_moves(board.legal_moves)
                for i in legals:
                    print(board.san(i), end='\t')
                
            else:
                print("Move is invalid! Try again.")

    if board.is_checkmate():
       print('Game over!\nYou Won!')
       break
   
    if draw:
        print("It is a draw!")
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
        print("It is a draw!")
        break