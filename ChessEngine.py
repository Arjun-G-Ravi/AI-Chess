import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

class NeuralNet(nn.Module):
    def __init__(self, inp_size=774, h1=500, h2=100, h3=100, out_size=1):
        super(NeuralNet, self).__init__()
        self.inp_size = inp_size
        self.lay1 = nn.Linear(inp_size, h1)
        self.lay2 = nn.ReLU()
        self.lay3 = nn.Linear(h1, h2)
        self.lay4 = nn.ReLU()
        self.lay5 = nn.Linear(h2, out_size)

    def forward(self,x):
        out = self.lay1(x)
        out = self.lay2(out)
        out = self.lay3(out) 
        out = self.lay4(out) 
        out = self.lay5(out) 
        return out

class DataSet(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape

    def __getitem__(self,indexVal):
        return self.x[indexVal], self.y[indexVal]
    
    def __len__(self):
        return self.m
   

class Engine:
    def __init__(self, board, model=None):
        self.board = board
        if model:
            
            self.model = joblib.load(model)
        else:
            self.model = None

    def get_fen(self, board):
        return board.fen()

    def encode_fen(self, fen_val):
        encoding = []
        fen_list = fen_val.split(' ')
        positions = fen_list[0].replace('/','')
        
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        for c in positions:
            chess_sq = [0 for i in range(12)] # one-hot
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
        
        # Castling oppurtunity?
        for c in ['K', 'Q', 'k', 'q']:
            if c in fen_list[2]:
                encoding.append(1)
            else:
                encoding.append(0)
    
        # En-passant oppurtunity?
        if fen_list[3] == '-':
            encoding.append(0) # Honestly, I have to write a better encoding to represent en-passant. Current encoding just shows if en-passant is available.
        else:
            encoding.append(1)
        
        return np.array(encoding,dtype=np.float32)
 
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

    def train_chess_engine(self, X, y, save=False, num_epochs=100, device='cuda'):
        model = NeuralNet().to(device) 
        lossCategory = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4)

        # Encoding
        X_encoded = torch.tensor([self.encode_fen(x) for x in X])
        y_encoded = torch.tensor([self.encode_y(i) for i in y])

        dataset = DataSet(X_encoded, y_encoded)
        train_loader = DataLoader(dataset=dataset, batch_size=10000, shuffle=True, num_workers=0)

        for epoch in range(num_epochs):
            for i,(x_mod, y_mod) in enumerate(train_loader):
                x_mod = x_mod.to(device)
                y_mod = y_mod.to(device)
                y_pred = model(x_mod).view(-1)
                loss = lossCategory(y_pred, y_mod)
                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                if ((i)%200) == 0:
                    print(f"Epoch:{epoch+1}/{num_epochs}  Loss:{loss*9999}") # Loss is de-normalised before displaying to user


        if save:
            import joblib
            joblib.dump(model,'Model_saves/ChessModel.joblib')
            
        self.model = model
        
    def mse(self, X, y):
        from sklearn.metrics import mean_squared_error
        y_pred =self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse
    
    def run_engine(self, X, model=None, device='cuda'):  
        ''' input: list
        output: float '''
        
        if model:
            import joblib
            self.model = joblib.load(model)
        # print(self.model)
        X_encoded = torch.tensor([self.encode_fen(x) for x in X]).to(device)
        # print(X_encoded)
            
        out = self.model(X_encoded)
        out = out  # de-normalise
        return out
    
    def accuracy(self, X, y,device='cuda:0'):
        
        X_encoded = torch.tensor([self.encode_fen(x) for x in X])
        y_encoded = torch.tensor([self.encode_y(i) for i in y])

        dataset = DataSet(X_encoded, y_encoded)
        test_loader = DataLoader(dataset=dataset, batch_size=10000, num_workers=0)
        lossCategory = nn.MSELoss()
        sum_correct, sum_loss = 0, 0
        
        for i,(x_mod, y_mod) in enumerate(test_loader):
            # print('banana')
            x_mod = x_mod.to(device)
            y_mod = y_mod.to(device)
            y_pred = self.model(x_mod).view(-1)
            sum_loss += lossCategory(y_pred, y_mod)
            sum_correct += torch.sum(torch.abs((y_mod - y_pred)*9999) < 100)
            
            
        print(f'Score: {sum_correct.item()}/{len(dataset)}, Accuracy: {sum_correct.item()/len(dataset)}')
        print('Avg_loss', sum_loss.item()/len(dataset))

        
        # print(f"""Score: {self.model.score(X_encoded,y_encoded):.3f}\n
        #           MSE  : {self.mse(X_encoded, y_encoded):.3f}\n{'-'*15}""")

if __name__ == '__main__':
    pass
    # Lets do engine training here
    # import pandas as pd
    # import numpy as np
    # import chess
    # from sklearn.model_selection import train_test_split
    # import joblib
 
    # # # Get dataset
    # df = pd.read_csv('/home/arjun/Desktop/Datasets/chessData.csv',nrows=100)
    # test_df = df.iloc[:100]
    # X = np.array(test_df.iloc[:,0])
    # y = np.array(test_df.iloc[:,1])
    # # dataset split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    
    # # # Initialisation
    # board = chess.Board()
    # engine = Engine(board)

    # # # Training
    # # print("Training model...")
    # # engine.train_chess_engine(X_train, y_train)
    
    # # Loading previous model
    # engine.model = joblib.load('Model_saves/Chess100kModel.joblib')

    # # Accuracy
    # engine.accuracy(X_train,y_train,1)
    # engine.accuracy(X_test,y_test,3)
    
    # # Inference
    # out = engine.run_engine(['rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2'])
    # print(out)