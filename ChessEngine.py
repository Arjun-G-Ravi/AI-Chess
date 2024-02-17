import chess
import numpy as np

class Engine:
    def __init__(self, board):
        self.board = board
        self.model = None

    def get_fen(self, board):
        return board.fen()
    
    '''
    def encode_fen(self, fen_val):
        # returns a vector with all encoded data of the board state. Now it is of length 70.
        encoding = []
        fen_list = fen_val.split(' ')
        # print(fen_list)

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
        return np.array(encoding)/12  # for normalising
    '''   
    def encode_fen(self, fen_val):
        encoding = []
        fen_list = fen_val.split(' ')
        positions = fen_list[0].replace('/','')
        
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        # if this doesnt work, i'll try one hot encoding
        for c in positions:
            chess_sq = [0 for i in range(12)]
            if c.isdigit():
                for _ in range(int(c)):
                    encoding.extend(chess_sq.copy())
            else:
                chess_sq[piece_hash_map[c]-1] = 1
                encoding.extend(chess_sq.copy())
                
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
                encoding.append(0)
    
        # En-passant oppurtunity
        if fen_list[3] == '-':
            encoding.append(0) # Honestly, I have to write a better encoding to represent en-passant. I'll do it if this thing works
        else:
            encoding.append(1)
        
        # print(encoding)
        return np.array(encoding,dtype=np.float32) # for normalising
 
    def encode_y(self, y):
        if '#+' in y:
            val = float(y.replace('#+',''))
            y = float(9999 - 100*(val-1))
        elif '#-' in y:
            val = float(y.replace('#-',''))
            y = float(-9999 + 100*(val-1))
        elif '+' in y or y == '0':
            y = float(y.replace('+',''))
        elif '-' in y:
            y = -float(y.replace('-',''))
        else:
            raise Exception('y Encoding Error')
        # print(y)
        return float(y/9999)  # for normalising 

    def train_chess_engine(self, X, y):
        from sklearn.neural_network import MLPRegressor
        import joblib

        model = MLPRegressor(solver='adam', max_iter=1000, hidden_layer_sizes=(774,1024,1024,512), learning_rate_init=1e-3)

        # Encoding
        X_encoded = np.array([self.encode_fen(x) for x in X])
        y_encoded = np.array([self.encode_y(i) for i in y])
        print(X_encoded[0])

        model.fit(X_encoded,y_encoded)
        # joblib.dump(model,'Model_saves/ChessModel.joblib' )
        self.model = model
        
    def mse(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred =self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse
    
    def run_engine(self, X, model=None):  
        ''' input: list
        output: float '''
        
        if model:
            import joblib
            self.model = joblib.load(model)  
        X_encoded = np.array([self.encode_fen(x) for x in X])
        out = self.model.predict(X_encoded)
        out = out*9999  # de-normalise
        return out
    
    def accuracy(self, X, y,type_=0):
        type_chart = {1:'Training set\n', 2:'Validation set\n', 3:'Test set\n'} 
        type_data ='' if not type_ else type_chart[type_]
        
        X_encoded = np.array([self.encode_fen(x) for x in X])
        y_encoded = np.array([self.encode_y(i) for i in y]) 
        print(f"""{'-'*15}\n{type_data}Score: {self.model.score(X_encoded,y_encoded):.3f}\nMSE  : {self.mse(X_encoded, y_encoded):.3f}\n{'-'*15}""")

if __name__ == '__main__':
    # Lets do engine training here
    import pandas as pd
    import numpy as np
    import chess
    from sklearn.model_selection import train_test_split
    import joblib
 
    # # Get dataset
    df = pd.read_csv('/home/arjun/Desktop/Datasets/chessData.csv',nrows=100)
    test_df = df.iloc[:100]
    X = np.array(test_df.iloc[:,0])
    y = np.array(test_df.iloc[:,1])
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    # # Initialisation
    board = chess.Board()
    engine = Engine(board)

    # # Training
    # print("Training model...")
    # engine.train_chess_engine(X_train, y_train)
    
    # Loading previous model
    engine.model = joblib.load('Model_saves/Chess100kModel.joblib')

    # Accuracy
    engine.accuracy(X_train,y_train,1)
    engine.accuracy(X_test,y_test,3)
    
    # Inference
    out = engine.run_engine(['rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'])
    print(out)