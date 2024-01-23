import chess
import chess.svg
import chess.engine
import random

piece_values = {
    chess.Piece.from_symbol('P'): 1,
    chess.Piece.from_symbol('N'): 3,
    chess.Piece.from_symbol('B'): 3,
    chess.Piece.from_symbol('R'): 5,
    chess.Piece.from_symbol('Q'): 9,
    chess.Piece.from_symbol('K'): 0,
    chess.Piece.from_symbol('p'): 1,
    chess.Piece.from_symbol('n'): 3,
    chess.Piece.from_symbol('b'): 3,
    chess.Piece.from_symbol('r'): 5,
    chess.Piece.from_symbol('q'): 9,
    chess.Piece.from_symbol('k'): 0
}

def countControlledSquares(board, color)
    controlled_squares = 0

    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            controlled_squares += 1

    return controlled_squares

def captureChooser(board):
    choice = None
    capture_value = 0
    squares_controlled = 0
    current_moves = board.legal_moves

    for move in current_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        elif capture_value == 0:
            if len(list(board.legal_moves)) > moves_max:
                choice = move
                moves_max = len(list(board.legal_moves))     
        board.pop()

        if board.is_en_passant(move):
            if capture_value == 0:
                choice = move
                capture_value = 1
        elif board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if piece_values[captured_piece] > capture_value:
                choice = move
                capture_value = piece_values[captured_piece]

    assert choice is not None, "captureChooser must choose a move."            
    return choice

STOCKFISH_LINUX = "/home/steven/Documents/stockfish/stockfish-ubuntu-x86-64-avx2"
STOCKFISH_MAC = "/opt/homebrew/Cellar/stockfish/16/bin/stockfish"

board = chess.Board()
for move in []:
    board.push_san(move)

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_LINUX)
print (engine)
move = captureChooser(board)

while not board.is_game_over():
    if board.turn:
        move = captureChooser(board)
    else:
        move = random.choice(list(board.legal_moves))

    #print (result.move)
    #board.push(result.move)
    #result = engine.play(board, chess.engine.Limit(time=0.1))

    print (move)
    board.push(move)


#chess.svg.board(board, size=500)
print (board)


engine.quit()