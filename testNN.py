import numpy as np
import chess
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

board = chess.Board()

model = MLPRegressor(solver='adam', max_iter=100, hidden_layer_sizes=(256,256,256))

X = np.array([1,2,3,4,5]).reshape(5,1)
y = [1,1,1,1,1]

model.fit(X,y)

score = model.score(X,y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(score,mse)



# import joblib
# import pandas as pd


# # Get dataset
# # df = pd.read_csv('/home/arjun/Desktop/Datasets/chessData.csv',nrows=5000)
# # test_df = df.iloc[:5000]
# # X = np.array(test_df.iloc[:,0])
# # y = np.array(test_df.iloc[:,1])
