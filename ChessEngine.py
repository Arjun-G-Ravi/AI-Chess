import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self, inp_size=774, h1=700, h2=700, h3=400, out_size=1):
        super(NeuralNet, self).__init__()
        self.inp_size = inp_size
        self.lay1 = nn.Linear(inp_size, h1)
        self.lay2 = nn.ReLU()
        self.lay3 = nn.Linear(h1, h2)
        self.lay4 = nn.ReLU()
        self.lay5 = nn.Linear(h2, h3)
        self.lay6 = nn.ReLU()
        self.lay7 = nn.Linear(h3, out_size)

    def forward(self,x):
        out = self.lay1(x)
        out = self.lay2(out)
        out = self.lay3(out) 
        out = self.lay4(out) 
        out = self.lay5(out) 
        out = self.lay6(out) 
        out = self.lay7(out) 
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
    def __init__(self, board, model=None, device='cuda'):
        self.board = board
        self.device=device
        if model:  self.model = torch.load(model)
        else: self.model = None

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
        try:
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
        except:
            print('Something wrong with', y)
            y = float(0)

        return float(y/9999)  # for normalising 

    def train_chess_engine(self, X, y, save=False, num_epochs=1000):
        print("Initialising...")
        model = NeuralNet().to(self.device) 
        lossCategory = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)

        # Encoding
        X_encoded = torch.tensor(np.array([self.encode_fen(x) for x in X]), dtype=torch.float32)
        y_encoded = torch.tensor(np.array([self.encode_y(i) for i in y]), dtype=torch.float32)

        dataset = DataSet(X_encoded, y_encoded)
        train_loader = DataLoader(dataset=dataset, batch_size=1500000, shuffle=True, num_workers=4)
        print("Training...")    
        for epoch in range(num_epochs):
            for i,(x_mod, y_mod) in enumerate(train_loader):
                x_mod = x_mod.to(self.device)
                y_mod = y_mod.to(self.device)
                y_pred = model(x_mod).view(-1)
                loss = lossCategory(y_pred, y_mod)
                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            print(f"Epoch: {epoch+1}/{num_epochs}   Loss: {loss*9999}") # Loss is de-normalised before displaying to user           
            if loss*9999 < 140 and loss*9999 > 120: torch.save(model,'Model_saves/ChessModel_120.pt')
            elif loss*9999 < 100 and loss*9999 > 90: torch.save(model,'Model_saves/ChessModel_90.pt')
            elif loss*9999 < 70 and loss*9999 > 60: torch.save(model,'Model_saves/ChessModel_60.pt')
            
            if loss*9999 < 50:
                 torch.save(self.model,'Model_saves/ChessModel_50.pt')
                 break
            
        self.model = model

    def run_engine(self, X, model=None):  
        if model:
            self.model = torch.load(model)
        X_encoded = torch.tensor(np.array([self.encode_fen(x) for x in X])).to(self.device)            
        return self.model(X_encoded)
    
    def accuracy(self, X, y,device='cuda:0'):

        X_encoded = torch.tensor(np.array([self.encode_fen(x) for x in X]), dtype=torch.float32)
        y_encoded = torch.tensor(np.array([self.encode_y(i) for i in y]), dtype=torch.float32)

        dataset = DataSet(X_encoded, y_encoded)
        test_loader = DataLoader(dataset=dataset, batch_size=600000, num_workers=4)
        lossCategory = nn.MSELoss()
        sum_correct, sum_loss = 0, 0
        
        for i,(x_mod, y_mod) in enumerate(test_loader):
            x_mod = x_mod.to(device)
            y_mod = y_mod.to(device)
            y_pred = self.model(x_mod).view(-1)
            sum_loss += lossCategory(y_pred, y_mod)
            sum_correct += torch.sum(torch.abs((y_mod - y_pred)*9999) < 100)
            
            
        print(f'Score: {sum_correct.item()}/{len(dataset)}, Accuracy: {sum_correct.item()*100/len(dataset):.2f}%')
        print('Avg_loss', sum_loss.item()/len(dataset))


if __name__ == '__main__':
    print('To train the model, run training_model.ipynb')