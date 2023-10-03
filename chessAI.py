import chess
import random

def get_legal_moves(generator_):
    l = []
    for i in generator_:
        l.append(i)
    return l

board = chess.Board()

game_over = board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves()

while not game_over:
    players_move = 1
    while players_move:
        try:
            move = input("\n\nYour move: ")
            board.push_san(move)
            players_move = 0
        except:
            print("Move is invalid")

    if game_over:
       break

    legal_moves = get_legal_moves(board.legal_moves)
    ai_move = random.choice(legal_moves)  # modify to be intelligent
    print(f'I will play: {ai_move}\n')
    board.push(ai_move)
    print(board)