import chess
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ChessDataSet(Dataset):
    def __init__(self, X, y):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __getitem__(self, indexVal):
        return self.x[indexVal], self.y[indexVal]
    
    def __len__(self):
        return len(self.y)
    
    def __repr__(self):
        return f'ChessDataSet Object <{format(len(self.y))}>'


class ChessNN(nn.Module):
    def __init__(self, inp_size, hidden1, hidden2, hidden3, out_size):
        super(ChessNN, self).__init__()
        self.inp_size = inp_size
        self.lay1 = nn.Linear(inp_size, hidden1)
        self.lay2 = nn.ReLU()
        self.lay3 = nn.Linear(hidden1, hidden2)
        self.lay4 = nn.ReLU()
        self.lay5 = nn.Linear(hidden2, hidden3)
        self.lay6 = nn.ReLU()
        self.lay7 = nn.Linear(hidden3, out_size)
        
    def forward(self, x):
        out = self.lay1(x) # maybe add float() here 
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        out = self.lay5(out)
        out = self.lay6(out)
        out = self.lay7(out) # We don't apply sotmax the cross entropy loss will do that for us
        return out


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
        return np.array(encoding)/12.0  # for normalising
    
    
    def encode_y(self, y):
        if '#+' in y:
            val = float(y.replace('#+',''))
            y = float(9999. - 100*(val-1))
        elif '#-' in y:
            val = float(y.replace('#-',''))
            y = float(-9999. + 100*(val-1))
        elif '+' in y or y == '0':
            y = float(y.replace('+',''))
        elif '-' in y:
            y = -float(y.replace('-',''))
        else:
            raise Exception('y Encoding Error')
        return float(y/9999.)  # for normalising 


    def train_chess_engine(self, X, y, num_epoch = 1, lr = 1e-5 ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        print(f'The device to be used for training: {device}')
        X = torch.tensor(np.array([self.encode_fen(x) for x in X]), dtype=torch.float32)
        print(X.shape)
        y = torch.tensor(np.array([self.encode_y(i) for i in y]), dtype=torch.float32)
        ds = ChessDataSet(X, y)
        dataloader = DataLoader(dataset=ds, batch_size=4, shuffle=True, num_workers=0)
        model = ChessNN(70, 100, 250, 100, 1). to(device)
        lossCategory = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        
        print("Training...")
        for epoch in range(num_epoch):
            for i,(X, y) in enumerate(dataloader):     
                X = X.to(device)
                y = y.to(device).reshape(-1,1)
                output = model(X).reshape(-1,1)
                loss = lossCategory(output, y)                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()                
            print(f"Epoch:{epoch} => Loss:{loss}")    
        return model

    def run_engine(self, X):  
        X_encoded = torch.tensor(np.array([self.encode_fen(x) for x in X]), dtype=torch.float32).to('cuda')
        print(X_encoded.shape)
        out = self.model(X_encoded)
        out = out*9999  # de-normalise
        print(out)
        return out
    
    def accuracy(self, X, y):
        y_pred = self.run_engine(X)
        print(y_pred.shape, y.shape)
        
        
if __name__ == '__main__':
    # Lets do engine training here
    import pandas as pd
    import numpy as np
    import chess
    from sklearn.model_selection import train_test_split
    import joblib
 
    # # Get dataset
    df = pd.read_csv('/home/arjun/Desktop/Datasets/chessData.csv',nrows=1000)
    test_df = df.iloc[:10000]
    X = np.array(test_df.iloc[:,0])
    y = np.array(test_df.iloc[:,1])
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    # # Initialisation
    board = chess.Board()
    engine = Engine(board)

    # # Training
    engine.model = engine.train_chess_engine(X_train, y_train)
    out = engine.run_engine(['rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2', 'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'])
    engine.accuracy(X_train,y_train)
    exit()  
    # Loading previous model
    # engine.model = joblib.load('Model_saves/Chess100kModel.joblib')

    # Accuracy
    engine.accuracy(X_test,y_test,3)
    
    # Inference
    print(out)