import chess

class Engine:
    def __init__(self, board):
        self.board = board

    def _get_fen(self):
        return self.board.fen()

    def encode_fen(self):
        # returns a vector with all encoded data of the board state. Now it is of length 70.
        encoding = []
        fen_val = self._get_fen()
        fen_list = fen_val.split(' ')

        # Position part
        positions = fen_list[0].replace('/','')
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        for c in positions:
            if c.isdigit():
                for i in range(int(c)):
                    encoding.append(0)
            else:
                encoding.append(piece_hash_map[c])

        # Whose move?
        if fen_list[1] == 'w':
            encoding.append(1)
        else:
            encoding.append(0)
        
        # Castling oppurtunity
        for c in ['K', 'Q', 'k', 'q']:
            if c in fen_list[2]:
                encoding.append(1)
            else:
                encoding.append(2)
    
        # En-passant oppurtunity
        if fen_list[3] == '-':
            encoding.append('0')  # Honestly, I have to write a better encoding to represent en-passant. I'll do it if this thing works
        else:
            encoding.append('1')

        return encoding
    
    
    def train_chess_engine(self, encoded_fen):
        pass

    def run_engine(self, model):
        pass

if __name__ == '__main__':
    board = chess.Board()
    engine = Engine(board)
    print(engine.encode_fen())

    