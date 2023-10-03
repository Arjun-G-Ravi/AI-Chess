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
            encoding.append(0)  # Honestly, I have to write a better encoding to represent en-passant. I'll do it if this thing works
        else:
            encoding.append(1)
        return encoding
    
    def encode_y(self,y):
        if '#+' in y:
            y = float(+999)
        elif '#-' in y:
            y = float(-999)
        elif '+' in y or y == '0':
            y = float(y.replace('+',''))
        elif '-' in y:
            y = -float(y.replace('-',''))
        else:
            print("Some error in ds")
        return float(y)   
    

    def train_chess_engine(self, X, y):
        from sklearn.neural_network import MLPClassifier
        import joblib

        model = MLPClassifier(solver='adam', max_iter=1000, alpha=1e-7, hidden_layer_sizes=(100,100,150,200,200,200,200,300,400,300,200,150,100,100,60,30))
        X_encoded = [self._encode_fen() for x in X]
        y_encoded = [self.encode_y(i) for i in y]
        print(X_encoded, y_encoded)
        model.fit(X_encoded,y_encoded)
        joblib.dump(model, 'chess_engine.pkl')
        print('Accuracy:',model.score(X_encoded,y_encoded)*100,'%')


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
    df = pd.read_csv('/home/arjun/Desktop/chessData.csv')
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
    input_fen =  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
    print(engine.run_engine(input_fen))

    