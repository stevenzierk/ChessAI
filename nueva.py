# See python-chess documentation at https://python-chess.readthedocs.io/en/latest/

import chess
import chess.pgn
import random
import time

# Plays a randomly selected legal move each turn
class RandomPlayer:
    def __init__(self):
        pass

    def select_move(self, board):
        return random.choice(list(board.legal_moves)) if board.legal_moves else None

# Plays a checkmate if possible, otherwise a capture if possible, 
# otherwise a random move
class CheckmateOrCapturePlayer:
    def __init__(self):
        pass

    def select_move(self, board):
        if not board.legal_moves:
            return None

        legal_moves = list(board.legal_moves)
        chosen_move = None

        for move in legal_moves:
            if board.is_capture(move):
                chosen_move = move
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        if not chosen_move:
            return random.choice(legal_moves)

        return chosen_move

# Plays a forced mate in up to 2 if possible, then prioritizes not being mated or stalemating,
# doing a capture if possible, otherwise a random move
class MateIn2OrCapturePlayer:
    def __init__(self):
        pass

    def find_captures(self, board):
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)

        return captures

    def find_mate(self, board):
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        return None

    def select_move(self, board):
        if not board.legal_moves:
            return None

        mate_avoided = list(board.legal_moves)

        maybe_mate = self.find_mate(board)
        if maybe_mate:
            return maybe_mate

        legal_moves = list(board.legal_moves)
        chosen_move = None
        for move in legal_moves:
            board.push(move)
            if board.is_stalemate() or self.find_mate(board): # opponent has mate
                mate_avoided.remove(move)
                board.pop()
                continue

            forced_mate = True
            for opp_move in list(board.legal_moves):
                board.push(opp_move)
                if not self.find_mate(board):
                    forced_mate = False
                    board.pop()
                    break
                board.pop()

            if forced_mate:
                board.pop()
                return move

            board.pop()

        captures = self.find_captures(board)
        captures = list(set(captures).intersection(mate_avoided))
        if len(captures) > 0:
            return random.choice(captures)

        if len(mate_avoided) > 0:
            return random.choice(mate_avoided)
        return random.choice(list(board.legal_moves))

def play_game(white, black):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        player = white if board.turn else black
        move = player.select_move(board)
        board.push(move)
        node = node.add_variation(move)

    game.headers["Result"] = board.result()
    game.headers["White"] = " " + type(white).__name__
    game.headers["Black"] = " " + type(black).__name__
    excluded_tags = ["Event", "Site", "Date", "Round"]
    for tag in excluded_tags:
        if tag in game.headers:
            del game.headers[tag]
    return game

def playoff(player1, player2):
    games = 100
    wins_1 = 0
    wins_2 = 0
    print ("Player 1: " + type(player1).__name__)
    print ("Player 2: " + type(player2).__name__)
    for i in range(games):
        #if (i + 1) & i == 0:
        #    print ("Games: " + str(i) + "  Player 1 wins: " + str(wins_1) + "  Player 2 wins: " + str(wins_2))
        if i % 2 == 0:
            white, black = player1, player2
        else:
            white, black = player2, player1

        game = play_game(white, black)
        result = game.headers["Result"]

        if (i % 2 == 0 and result == "1-0") or (i % 2 == 1 and result == "0-1"):
            wins_1 += 1
        elif result != "1/2-1/2":
            wins_2 += 1

    score = wins_1 + (100 - wins_1 - wins_2) * .5

    print(str(wins_1) + " wins for Player 1, " + str(100 - wins_1 - wins_2) + " draws, and " + str(wins_2) + " wins for Player 2.")
    print("Total match score: " + str(score) + " - " + str(100 - score))

startTime = time.time()

board = chess.Board()
game = chess.pgn.Game()
node = game

rng = RandomPlayer()
m1 = CheckmateOrCapturePlayer()
m2 = MateIn2OrCapturePlayer()

player1 = rng
player2 = m2
playoff(player1, player2)

print ("Runtime: %.2f" % (time.time() - startTime) + " seconds.")
