import chess
import chess.svg
from PIL import Image, ImageTk
import io
import tkinter as tk

def draw_board(board):
    # Generate SVG of the board
    svg = chess.svg.board(board)
    svg_bytes = svg.encode('utf-8')

    # Convert SVG to an image that Tkinter can use
    with io.BytesIO(svg_bytes) as svg_image:
        with Image.open(svg_image) as image:
            return ImageTk.PhotoImage(image)

def make_move():
    try:
        # Make a move (for this example, we just make a random move)
        move = next(iter(board.legal_moves))
        board.push(move)
        # Update the board image
        board_image = draw_board(board)
        board_label.config(image=board_image)
        board_label.image = board_image # Keep a reference!
    except StopIteration:
        pass  # No more legal moves

# Initialize chess board
board = chess.Board()

# Create the main window
root = tk.Tk()
root.title('Chess Board')

# Draw the initial board
board_image = draw_board(board)
board_label = tk.Label(root, image=board_image)
board_label.pack()

# Button to make a move
move_button = tk.Button(root, text='Make a Move', command=make_move)
move_button.pack()

root.mainloop()