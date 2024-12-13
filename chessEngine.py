import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from print_color import print

class ChessEncoder:

    def encode_fen(self, fen):
        fen = fen.split(' ')[:4]
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
        encoding.extend(white_encoding)
        encoding.extend(black_encoding)
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
    def __init__(self, embedding_dim=1, bs_train=1, bs_eval=1):
        super(MLPEngine, self).__init__()
        self.bs_train = bs_train
        self.bs_eval = bs_eval
        self.embd1 = nn.Embedding(200, embedding_dim)
        
        self.l1 = nn.Linear(200*embedding_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.l2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.l3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.l5 = nn.Linear(256, 1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        
        
        # Weight initialization
        torch.nn.init.kaiming_uniform_(self.l1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.l2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.l3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_uniform_(self.embd1.weight, nonlinearity='leaky_relu')

    
    def forward(self, x):
        # print(x.shape, color='b')
        out = self.embd1(x)
        # print('cow 1')
        if not self.training:
            # print(out.shape)
            out = torch.flatten(out, start_dim=1)
            # print(out.shape)
            out = out.view(self.bs_eval, -1)
            # print(out.shape)
        else:
            out = torch.flatten(out, start_dim=1)
            out = out.view(self.bs_train, -1)
        

        # print('cow 2')
        # if self.training:
        # else:
            
        
        # print('cow 2')
        # print(out.shape, color='m')
        out = F.leaky_relu(self.bn1(self.l1(out)))    
        # print('cow 22')
        out = self.dropout1(out)
        # print('cow 23')
        out = F.leaky_relu(self.bn2(self.l2(out)))
        
        # print('cow 3')
        out = self.dropout2(out)
        out = F.leaky_relu(self.bn3(self.l3(out)))
        
        # print('cow 4')

        # print('cow 5')
        out = self.l5(out)
        return out

def _get_score(save_path, input_fen):
    # encoding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_object = ChessEncoder()
    encoded_fen = torch.tensor(encoder_object.encode_fen(input_fen), dtype=torch.int32).to(device).view(200)
    bs=1
    model = MLPEngine(embedding_dim=64).to(device)
    model.load_state_dict(torch.load('saves/bad_model.pt', weights_only=True))
    model.eval()
    y_pred = model(encoded_fen)
    return y_pred.item()

def get_best_move(board, save_path):
    best_move = ''
    best_move_score = 1000
    for possible_move in board.legal_moves:
        updated_board = deepcopy(board)
        updated_board.push(possible_move)
        current_score = _get_score(save_path, updated_board.fen())
        if current_score < best_move_score:
            # this is the most negative move, which is good for black
            best_move_score = current_score
            best_move = possible_move
    return best_move


if __name__ == '__main__':
    # c = ChessEncoder()

    # x = c.encode_fen('rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
    # X = []
    # for i in range(16):
    #     X.append(x)
    # y = c.encode_score('-33')
    # X = torch.tensor(X, dtype=torch.int16)
    # y = torch.tensor(y)
    # print(X, X.shape, y)
    score = _get_score('saves/bad_model.pt', 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1') 
    board = chess.Board()
    print(get_best_move(board, 'saves/bad_model.pt'))
