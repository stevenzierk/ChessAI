import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

piece_values = {
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
    'p': -1,
    'n': -2,
    'b': -3,
    'r': -4,
    'q': -5,
    'k': -6
}

# Takes a chess position in FEN and creates the following 71 value list:
# 64 values for the occupant of each square
# 1 value for the player to move (White = 1, Black = 0)
# 4 values for possible castling (1 - yes, 0 - no)
# 1 value for if en passant on a file is possible (none - 0, if so, 1-8)
# 1 value for ply toward the fifty move rule (up to 99)
def fen_to_nninput(fen):
    nninput = []
    data = fen.split(" ")

    for char in data[0]:
        if char == '/':
            continue
        elif char.isdigit():
            nninput.extend([0] * int(char))
        elif char in piece_values:
            nninput.append(piece_values[char])
        else:
            raise ValueError("Unexpected character in FEN string")

    #assert len(nninput) == 64, "Error: expected 64 square values but have " + str(len(nninput))

    move = 1 if data[1] == 'w' else 0
    nninput.append(move)

    for letter in "KQkq":
        nninput.append(1 if letter in data[2] else 0)

    assert len(nninput) == 69

    nninput.append(ord(data[3][0]) - 97 + 1 if data[3][0] in "abcdefgh" else 0)

    nninput.append((int)(data[4]))
    assert len(nninput) == 71

    return nninput

piece_values2 = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11
}

def fen_to_nninput2(fen):
    piece_channels = np.zeros((12, 8, 8))
    data = fen.split(" ")

    row, col = 0, 0
    for char in data[0]:
        if char == '/':
            col = 0
            row += 1
        elif char.isdigit():
            col += int(char)
        elif char in piece_values2:
            piece_channels[piece_values2[char]][row][col] = 1
            col += 1

    move_channel = np.zeros((8, 8)) if data[1] == 'w' else np.ones((8, 8))

    castle_channels = np.zeros((4, 8, 8))
    letters = "KQkq"
    for i in range(4):
        if letters[i] in data[2]:
            castle_channels[i].fill(1)

    return np.vstack([piece_channels, move_channel[np.newaxis, :, :], castle_channels])

def pick_move_depth_1(mover, board):
    best = -1
    multiplier = 1 if board.turn == chess.WHITE else -1
    picked = None
    for move in board.legal_moves:
        board.push(move)
        data = fen_to_nninput(board.fen())
        data = torch.tensor(data, dtype=torch.float32)
        score = mover(data)
        if score * multiplier > best:
            picked = move
            best = score
        board.pop()

    return (picked, score)

def nn_eval(board, nn):
    data = fen_to_nninput(board.fen())
    data = torch.tensor(data, dtype=torch.float32)
    return nn(data)

def minimax(board, depth, is_maximizer, nn, alpha=-1, beta=1):
    if depth == 0 or board.is_game_over():
        return nn_eval(board, nn)

    if is_maximizer:
        max_eval = -1
        for move in board.legal_moves:
            board.push(move)
            move_eval = minimax(board, depth-1, False, nn, alpha, beta)
            board.pop()
            max_eval = max(max_eval, move_eval)
            alpha = max(max_eval, move_eval)
            if alpha >= beta:
                break
        return max_eval
    else:
        min_eval = 1
        for move in board.legal_moves:
            board.push(move)
            move_eval = minimax(board, depth-1, True, nn, alpha, beta)
            board.pop()
            min_eval = min(min_eval, move_eval)
            beta = min(beta, move_eval)
            if beta <= alpha:
                break
        return min_eval

def select_best_move(board, depth, nn):
    best_move = None
    best_eval = -1 if board.turn == chess.WHITE else 1

    for move in board.legal_moves:
        board.push(move)
        move_eval = minimax(board, depth - 1, board.turn == chess.BLACK, nn)
        board.pop()

        if (board.turn == chess.WHITE and move_eval > best_eval) or (board.turn == chess.BLACK and move_eval < best_eval):
            best_eval = move_eval
            best_move = move

    return (best_move, best_eval)

def self_play(nn, depth=1):
    board = chess.Board()
    positions = []

    while not board.is_game_over():
        positions.append(board)
        move, score = select_best_move(board, depth, nn)
        board.push(move)
    return positions, board.result()

def play_game(net1, net2, depth=1):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        net = net1 if board.turn == chess.WHITE else net2
        data = fen_to_nninput(board.fen())
        data = torch.tensor(data, dtype=torch.float32)
        move, score = select_best_move(board, depth, net)
        print (move, score)
        board.push(move)
        node = node.add_variation(move)

class ChessEvaulationNet1(nn.Module):

    def __init__(self):
        super(ChessEvaulationNet1, self).__init__()
        self.fc1 = nn.Linear(71, 71)
        self.fc2 = nn.Linear(71, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

def nn_eval2(board, nn):
    data = fen_to_nninput2(board.fen())
    data = torch.tensor(data, dtype=torch.float32)
    return nn(data)

def select_best_move2(board, nn):
    best_move = None
    best_eval = -1 if board.turn == chess.WHITE else 1

    if not board.legal_moves:
        print ("uhoh")
        return None, 0

    for move in board.legal_moves:
        board.push(move)
        move_eval = nn_eval2(board, nn)
        board.pop()

        if (board.turn == chess.WHITE and move_eval >= best_eval) or (board.turn == chess.BLACK and move_eval <= best_eval):
            best_eval = move_eval
            best_move = move

    if best_move is None:
        print ("No best move, best eval: " + best_eval)

    return (best_move, best_eval)

def self_play2(nn):
    board = chess.Board()
    positions = []

    while not board.is_game_over():
        positions.append(board)
        move, score = select_best_move2(board, nn)
        board.push(move)
    return positions, board.result()

class ChessCNN(nn.Module):

    def __init__(self):
        super(ChessCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 128 * 2 * 2)

        # Apply fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Output between -1 and 1
        x = torch.tanh(x)
        return x

conv_chess_net = ChessCNN()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(conv_chess_net.parameters(), lr=0.001)

for i in range(1, 100 + 1):
    print (f"Game {i}")

    positions, result = self_play2(conv_chess_net)

    outcome = 1 if result == "1-0" else -1 if result == "0-1" else 0
    outcome = torch.tensor([[outcome]], dtype=torch.float32)

    for epoch in range(10):
        for position in positions:
            conv_chess_net.train()
            optimizer.zero_grad()
            prediction = nn_eval2(position, conv_chess_net)

            loss = loss_function(prediction, outcome)
            loss.backward()
            optimizer.step()

conv_chess_net.eval()
self_play2(conv_chess_net)

board = chess.Board()
game = chess.pgn.Game()
node = game
game.headers["Event"] = "NN eval depth 1 game"
white = conv_chess_net
black = ChessCNN()
white.eval()
black.eval()

while not board.is_game_over():
    with torch.no_grad():
        mover = white if board.turn == chess.WHITE else black
        move, score = select_best_move2(board, mover)
        board.push(move)
        node = node.add_variation(move)

print (board)
game.headers["Result"] = board.result()
print (str(game))

'''
net1 = ChessEvaulationNet1
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net1.parameters(), lr=0.001)

for i in range(1, 10 + 1):
    net1.eval()
    #if i & (i - 1) == 0:
    #    print (i)
    print (i)

    positions, result = self_play(net1, 3)

    outcome = 0
    if result == "1-0":
        outcome = 1
    elif result == "0-1":
        outcome = -1
    outcome = torch.tensor([outcome], dtype=torch.float32)

    for epoch in range(1):
        for position in positions:
            net1.train()
            optimizer.zero_grad()
            prediction = nn_eval(position, net1)

            loss = loss_function(prediction, outcome)
            loss.backward()
            optimizer.step()

net1.eval()
net2 = ChessEvaulationNet1()
net2.eval()

play_game(net1, net2, 3)
input_start_pos = fen_to_nninput(chess.STARTING_FEN)
input_start_pos = torch.tensor(input_start_pos, dtype=torch.float32)
net1 = ChessEvaulationNet1()
net1.eval()
with torch.no_grad():
    output = net1(input_start_pos)

print(output)'''

'''board = chess.Board()
game = chess.pgn.Game()
node = game
game.headers["Event"] = "NN eval depth 1 game"
white = ChessEvaulationNet1()
black = ChessEvaulationNet1()
white.eval()
black.eval()

while not board.is_game_over():
    with torch.no_grad():
        mover = white if board.turn == chess.WHITE else black
        move, score = pick_move_depth_1(mover, board)
        board.push(move)
        node = node.add_variation(move)

print (board)
game.headers["Result"] = board.result()
print (str(game))'''









