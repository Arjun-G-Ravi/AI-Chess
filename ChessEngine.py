import chess
# This is new os 2
# This is latest branch

class Engine:
    def __init__(self, board):
        self.board = board
        self.model = None

    def get_fen(self, board):
        return board.fen()

    def encode_fen(self, fen_val):
        # returns a vector with all encoded data of the board state. Now it is of length 70.
        encoding = []
        fen_list = fen_val.split(' ')

        # Position part
        positions = fen_list[0].replace('/','')
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        # if this doesnt work, i'll try one hot encoding
        for c in positions:
            if c.isdigit():
                for _ in range(int(c)):
                    encoding.append(float(0))
            else:
                encoding.append(float(piece_hash_map[c]))

        # Whose move?
        if fen_list[1] == 'w':
            encoding.append(1.)
        else:
            encoding.append(0.)
        
        # Castling oppurtunity
        for c in ['K', 'Q', 'k', 'q']:
            if c in fen_list[2]:
                encoding.append(1.)
            else:
                encoding.append(0.)
    
        # En-passant oppurtunity
        if fen_list[3] == '-':
            encoding.append(0.)  
        # Honestly, I have to write a better encoding to represent en-passant. I'll do it if this thing works
        else:
            encoding.append(1.)
        return encoding
    
    def encode_y(self, y):
        if '#+' in y:
            y = float(+9999)
        elif '#-' in y:
            y = float(-9999)
        elif '+' in y or y == '0':
            y = float(y.replace('+',''))
        elif '-' in y:
            y = -float(y.replace('-',''))
        else:
            raise Exception('y Encoding Error')
        return float(y)
    
    def normalise(self, inp):
        inp_mean = np.mean(inp)
        inp_std = np.std(inp)
        inp_norm = [(i - inp_mean) / inp_std for i in inp]
        return inp_norm


    def train_chess_engine(self, X, y):
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(solver='adam', max_iter=1000, hidden_layer_sizes=(512,512,512,512))
        
        X_fen = [self.get_fen(self.board) for x in X]

        # Encoding
        X_encoded = np.array([self.encode_fen(x) for x in X_fen])
        y_encoded = np.array([self.encode_y(i) for i in y])

        # Normalising
        # X_norm = self._normalise(X_encoded)
        # y_norm = self._normalise(y_encoded)

        model.fit(X_encoded,y_encoded)
        self.model = model
        print(f"{'-'*30}\nScore: {model.score(X_encoded,y_encoded)}\n{'-'*30}")
        
        # from sklearn.metrics import mean_squared_error
        # y_pred = self.model.predict(X_encoded)
        # print(y_pred)


    def run_engine(self, X):
        X_fen = [self.get_fen(self.board) for x in X]
        X_encoded = np.array([self.encode_fen(x) for x in X_fen])
        out = self.model.predict(X_encoded)
        # I will de-normalise the output after correcting the accuracy of the model, which is 0 right now
        return out


if __name__ == '__main__':
    # Lets do engine training here
    import pandas as pd
    import numpy as np
    import chess

    # Get dataset
    df = pd.read_csv('/home/arjun/Desktop/Datasets/chessData.csv',nrows=5000)
    test_df = df.iloc[:2]
    X = np.array(test_df.iloc[:,0])
    y = np.array(test_df.iloc[:,1])

    # Initialisation
    board = chess.Board()
    engine = Engine(board)

    # Training
    print("Training model...")
    engine.train_chess_engine(X, y)

    # Inference
    out = engine.run_engine(X)
    print(out)