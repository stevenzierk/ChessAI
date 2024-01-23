import random

class MateIn2OrCapturePlayer:
    def __init__(self):
        pass

    def find_captures(board):
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)

        return captures

    def find_mate(board):
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        return None

    def select_move(board):
        if not board.legal_moves:
            return None

        maybe_mate = find_mate(board)
        if maybe_mate:
            return maybe_mate

        legal_moves = list(board.legal_moves)
        chosen_move = None
        for move in legal_moves:
            board.push(move)
            if find_mate(board): # opponent has mate
                board.pop()
                continue

            forced_mate = True
            for opp_move in list(board.legal_moves):
                board.push(opp_move)
                if not find_mate(board):
                    forced_mate = False
                    board.pop()
                    break
                board.pop()

            if forced_mate:
                board.pop()
                return move

            board.pop()

        captures = find_captures(board)
        if len(captures) > 0:
            return random.choice(captures)

        return random.choice(list(board.legal_moves))


