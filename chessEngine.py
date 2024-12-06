import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
class ChessEncoder:

    def encode_fen(self, fen):
        fen = fen.split(' ')[:4]
        print(fen)
        encoding = []
        white_encoding = []
        black_encoding = []
        tot_piece_value_white = 0
        tot_piece_value_black = 0
        piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k':0, 'K':0}
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        for i in fen[0].replace('/', ''):
            if not i.isdigit():
                encoding.append(piece_hash_map[i])
                if i.isupper():
                    white_encoding.append(piece_hash_map[i])
                    tot_piece_value_white += piece_values[i]
                    black_encoding.append(0)
                elif not i.isupper():
                    black_encoding.append(piece_hash_map[i])
                    white_encoding.append(0)
                    tot_piece_value_black += -piece_values[i]
                else:
                    raise 'error here 1'
            elif i.isdigit():
                for _ in range(int(i)):
                    encoding.append(0)
                    white_encoding.append(0)
                    black_encoding.append(0)
            else: 
                print('Something wierd here!')
                raise 'error here 2'
        # print(len(encoding), len(white_encoding), len(black_encoding))
        # print(white_encoding)
        encoding.extend(white_encoding)
        encoding.extend(black_encoding)
        # print(len(encoding))
        assert len(encoding) == 64*3
        
        # Whose move?
        if fen[1] == 'w':
            encoding.append(1)
        else:
            encoding.append(0)
        
        # Castling oppurtunity
        for c in ['K', 'Q', 'k', 'q']:
            if c in fen[2]:
                encoding.append(1)
            else:
                encoding.append(0)

        # En passant availability
        if fen[3] == '-':
            encoding.append(0.)  
        else:
            encoding.append(1.)

        # encode piece values
        encoding.append(tot_piece_value_white)
        encoding.append(tot_piece_value_black)

        assert(len(encoding) == 200)
        # print(encoding)
        return encoding

    def encode_score(self, score):
        if '#+' in score:
            val = float(score.replace('#+',''))
            score = float(9999 - 100*(val-1))
        elif '#-' in score:
            val = float(score.replace('#-',''))
            score = float(-9999 + 100*(val-1))
        elif '+' in score or score == '0':
            score = float(score.replace('+',''))
        elif '-' in score:
            score = -float(score.replace('-',''))
        return float(score)
    
class MLPEngine(nn.Module):
    def __init__(self, embedding_dim=32):
        super(MLPEngine, self).__init__()
        self.embd1 = nn.Embedding(200,  embedding_dim)
        self.l1 = nn.Linear(200*embedding_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.l2 = nn.Linear(1024, 128)
        self.ln2 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        torch.nn.init.kaiming_uniform_(self.l1.weight)
        torch.nn.init.kaiming_uniform_(self.l2.weight)
        torch.nn.init.xavier_uniform_(self.l3.weight)
        torch.nn.init.kaiming_uniform_(self.embd1.weight)

    
    def forward(self, x):
        out = self.embd1(x)
        # print(out.shape)
        if not self.training:
            out = torch.flatten(out)
        else:
            out = torch.flatten(out, start_dim=1)
        # print(out.shape)
        out = F.leaky_relu(self.ln1(self.l1(out)))    
        out = self.dropout1(out)
        out = F.leaky_relu(self.ln2(self.l2(out)))
        # out = self.dropout2(out)
        out = self.l3(out)
        return out


if __name__ == '__main__':
    c = ChessEncoder()
    x = c.encode_fen('rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
    X = []
    for i in range(16):
        X.append(x)
    y = c.encode_score('-33')
    X = torch.tensor(X, dtype=torch.int16)
    y = torch.tensor(y)
    print(X, X.shape, y)
    
