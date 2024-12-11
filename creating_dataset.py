with open('data/chessData.csv') as f:
    data1 = f.read().split('\n')


with open('data/fen_analysis.csv') as f:
    data2 = f.read().split('\n')

with open('data/random_evals.csv') as f:
    data3 = f.read().split('\n')

with open('data/tactic_evals.csv') as f:
    data4 = f.read().split('\n')

data1.extend(data2)
data1.extend(data3)
data1.extend(data4)
print(len(data1))
import pandas as pd

col1 = []
col2 = []
print(data1[:10])
for data in data1:
    if data.split(',')[0] and data.split(',')[1]:
        i = data.split(',')[0]
        v = data.split(',')[1]
        if i == 'FEN' or not i or i == 'fen_value':
            continue

        col1.append(i)
        col2.append(v)

print(len(col1), len(col2))
df = pd.DataFrame({'fen':col1, 'score':col2}).sample(frac=1)
df.to_csv('data/chessDataFinal.csv', index=False)
print('Saved')