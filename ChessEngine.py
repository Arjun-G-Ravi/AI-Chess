import chess

class Engine:
    def __init__(self, board):
        self.board = board

    def _get_fen(self):
        return self.board.fen()

    def _encode_fen(self):
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
                encoding.append(2.)
    
        # En-passant oppurtunity
        if fen_list[3] == '-':
            encoding.append(0.)  # Honestly, I have to write a better encoding to represent en-passant. I'll do it if this thing works
        else:
            encoding.append(1.)
        return encoding
    
    def encode_y(self,y):
        if '#+' in y:
            y = float(+9999)
        elif '#-' in y:
            y = float(-9999)
        elif '+' in y or y == '0':
            y = float(y.replace('+',''))
        elif '-' in y:
            y = -float(y.replace('-',''))
        else:
            print("Some error in ds")

        # normalise y
        

        return float(y)
    

    def train_chess_engine(self, X, y):
        from sklearn.neural_network import MLPRegressor
        import joblib

        model = MLPRegressor(solver='adam', max_iter=30000, learning_rate_init=1e-4, alpha=1e-7, hidden_layer_sizes=(256,512,1024,1024,256,256))
        X_encoded = [self._encode_fen() for x in X]
        y_encoded = [self.encode_y(i) for i in y]

        # Normalising
        X_mean = np.mean(X_encoded)
        X_std = np.std(X_encoded)
        X_norm = [(i - X_mean) / X_std for i in X_encoded]
        
        y_mean = np.mean(y_encoded)
        y_std = np.std(y_encoded)
        y_norm = [(i - y_mean) / y_std for i in y_encoded]

        # print([(i,j) for i,j in zip(X_norm, y_norm)])

        model.fit(X_norm,y_norm)
        joblib.dump(model, 'chess_engine.pkl')
        print('Score: ',model.score(X_norm,y_norm))


    def run_engine(self, X):
        import joblib
        import numpy as np
        model = joblib.load('chess_engine.pkl')

        new_board = chess.Board()
        encodings = Engine(new_board)._encode_fen()
        # print(encodings)
        return model.predict([encodings])


if __name__ == '__main__':
    # Lets do engine training here
    # ----------------------------
    import pandas as pd
    import numpy as np
    import chess
    
    # Get dataset
    print("Loading dataset ...")
    df = pd.read_csv('/home/arjun/Desktop/chessData.csv',nrows=1001)
    test_df = df.iloc[:1000]
    X = np.array(test_df.iloc[:,0])
    y = np.array(test_df.iloc[:,1])

    # Initialisation
    board = chess.Board()
    engine = Engine(board)

    # Training
    print("Training model...")
    engine.train_chess_engine(X, y)

    # Inference
    input_fen = X[1]
    print(X[1],y[1])
    print(engine.run_engine(input_fen))

    