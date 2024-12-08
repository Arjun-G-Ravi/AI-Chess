# auto queen happens when you promote

from print_color import print
import pygame
import sys
import chess
import time
from chessEngine import get_best_move

pygame.init()

WIDTH, HEIGHT = 800, 800  # Window size
ROWS, COLS = 8, 8  # Chessboard grid
SQUARE_SIZE = WIDTH // COLS
WHITE, BLACK = (240, 217, 181), (181, 136, 99)  # Chessboard colors
save_path = 'saves/bad_model.pt'

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")

pieces = {}
piece_mapping = {
    (chess.PAWN, chess.WHITE): "wP",
    (chess.PAWN, chess.BLACK): "bP",
    (chess.ROOK, chess.WHITE): "wR",
    (chess.ROOK, chess.BLACK): "bR",
    (chess.KNIGHT, chess.WHITE): "wN",
    (chess.KNIGHT, chess.BLACK): "bN",
    (chess.BISHOP, chess.WHITE): "wB",
    (chess.BISHOP, chess.BLACK): "bB",
    (chess.QUEEN, chess.WHITE): "wQ",
    (chess.QUEEN, chess.BLACK): "bQ",
    (chess.KING, chess.WHITE): "wK",
    (chess.KING, chess.BLACK): "bK"
}

# Preload and scale piece images
for (piece_type, color), filename in piece_mapping.items():
    try:
        pieces[(piece_type, color)] = pygame.transform.scale(
            pygame.image.load(f"./pieces/{filename}.svg"), (SQUARE_SIZE, SQUARE_SIZE)
        )
    except Exception as e:
        print(f"Could not load image for {filename}: {e}")

board = chess.Board()

def draw_board():
    """Draw the chessboard."""
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    """Draw chess pieces on the board."""
    for row in range(ROWS):
        for col in range(COLS):
            # Convert board coordinates to chess square notation
            square = chess.square(col, 7 - row)  # Flip the row to match chess board orientation
            piece = board.piece_at(square)
            
            if piece:
                if piece == 'P':
                    print('BANANa')
                # Get the corresponding piece image
                try:
                    screen.blit(
                        pieces[(piece.piece_type, piece.color)], 
                        (col * SQUARE_SIZE, row * SQUARE_SIZE)
                    )
                except KeyError:
                    print(f"No image found for piece type {piece.piece_type}, color {piece.color}")

def get_square_from_mouse(pos):
    """Convert mouse position to chess square coordinates."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = 7 - (y // SQUARE_SIZE)  # Flip row to match chess board orientation
    return chess.square(col, row)

def main():
    selected_square = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                clicked_square = get_square_from_mouse(event.pos)
                
                if selected_square is None: # if a piece is selected for movement
                    if board.piece_at(clicked_square):
                        selected_square = clicked_square
                else:
                    try:
                        move = chess.Move(selected_square, clicked_square)
                        move_if_promotion = chess.Move(selected_square, clicked_square, 5) # queen

                        if move in board.legal_moves or move_if_promotion in board.legal_moves:
                            if move in board.legal_moves:
                                board.push(move)
                                draw_board()
                                draw_pieces()
                                pygame.display.flip()
                            else: # This happens only if there is a promotion present, we auto-queen
                                board.push(move_if_promotion)
                            
                            if not board.is_game_over():
                                
                                # Choosing the first move in the list to play as Computer
                                # computer_move = chess.Move.null()
                                # for possible_move in board.legal_moves:
                                #     computer_move = possible_move
                                #     break

                                # When AI will be ready, we go:
                                computer_move = get_best_move(board, save_path)
                                board.push(computer_move)
                                if board.is_checkmate():
                                    draw_board()
                                    draw_pieces()
                                    pygame.display.flip()
                                    print('It is a checkmate. The Computer wins!')
                                    time.sleep(3)
                                    pygame.quit()
                                    sys.exit()
                            else: # game is over
                                # The game is somehow over
                                if board.is_checkmate():
                                    draw_board()
                                    draw_pieces()
                                    pygame.display.flip()
                                    print('It is a checkmate. You win!')
                                else:
                                    draw_board()
                                    draw_pieces()
                                    pygame.display.flip()
                                    print("The game is a draw")
                                time.sleep(3)
                                pygame.quit()
                                sys.exit()
                        
                        else:
                            print('Illegal move')
                        selected_square = None
                    
                    except Exception as e:
                        print(f"Invalid move: {e}")
                        selected_square = None
        
        draw_board()
        draw_pieces()
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()