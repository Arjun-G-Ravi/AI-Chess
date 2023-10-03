import chess

class Engine:
    def __init__(self, board):
        self.board = board

    def get_fen(self, board):
        return board.fen()

    def encode_fen(self, fen_board):
        pass

    def train_chess_engine(self, encoded_fen):
        pass

    def run_engine(self, model, board):
        pass

if __name__ == '__main__':
    print("cow")