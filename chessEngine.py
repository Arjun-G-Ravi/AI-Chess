import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class chessEngine:
    def __init__(self):
        pass

    def encode_fen(self, fen):
        fen = fen.split(' ')[:4]
        encoding = []
        piece_hash_map = {'K':1, 'Q':2, 'R':3, 'B':4, 'N':5, 'P':6, 'k':7, 'q':8, 'r':9, 'b':10, 'n':11, 'p':12}
        for i in fen[0].replace('/', ''):
            if not i.isdigit():
                encoding.append(piece_hash_map[i])
            elif i.isdigit():
                for _ in range(int(i)):
                    encoding.append(0)
            else: 
                print('Something wierd here!')
        assert len(encoding) == 64
        
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

        if fen[3] == '-':
            encoding.append(0.)  
        else:
            encoding.append(1.)
        assert(len(encoding) == 70)
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
        else:
            raise Exception('y Encoding Error')
        # print(float(score/9999))
        return float(score/9999)
    
    def train(self, X, y):

        
        pass


class MLP(nn.Module):
    def __init__(self):
        


if __name__ == '__main__':
    c = chessEngine()
    x = c.encode_fen('rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
    X = []
    for i in range(16):
        X.append(x)
    y = c.encode_score('-33')
    X = torch.tensor(X, dtype=torch.int16)
    y = torch.tensor(y)
    print(X, X.shape, y)
    
