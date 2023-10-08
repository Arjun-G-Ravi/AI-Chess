import chess
import ChessEngine

b = chess.Board()
engine = ChessEngine.Engine(b)

print(engine.run_engine(['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/QNBQKQQQ w KQkq - 0 0'],'Model_saves/100KChess_64.joblib'))
