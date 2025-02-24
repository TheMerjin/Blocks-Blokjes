import numpy as np
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Initialize Pygame
pygame.init()

# Constants for the game
CELL_SIZE = 100  # Size of each cell in pixels
BOARD_SIZE = 5   # 5x5 board
NEXT_PIECES_COUNT = 3  # Number of next pieces to display
MARGIN = 20  # Margin between board and next pieces display


# Calculate window dimensions
BOARD_WIDTH = 100 * BOARD_SIZE
NEXT_PIECES_WIDTH = CELL_SIZE * 4  # Width allocated for next pieces
WINDOW_WIDTH = BOARD_WIDTH + MARGIN + NEXT_PIECES_WIDTH
WINDOW_HEIGHT = 100 * BOARD_SIZE

# Set up display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pygame Board Game with Next Pieces")
font = pygame.font.SysFont(None, 24)

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

class Game:
    def __init__(self):
        """Initializes the game board and other attributes."""
        self.board = np.zeros((5, 5))  # Empty 5x5 board
        self.pts_per_move = 1  # Points per valid move
        self.score = 0  # Current score
        self.held_piece = None  # No piece held initially
        self.current_piece = self.generate_random_piece()  # Generate the first piece
        self.next_pieces = [self.generate_random_piece() for _ in range(11)]
        self.move_list = []
    def hold_piece(self):
        if self.held_piece is None:
            self.held_piece =   self.current_piece
            self.current_piece = self.next_pieces.pop(0)  # Get the next piece
            self.next_pieces.append(self.generate_random_piece())
    def unhold_piece(self):
        self.next_pieces.insert(0,self.current_piece)
        self.current_piece = self.held_piece
        self.held_piece = None
    def play_move(self, move):
        if move.hold == True:
            self.hold()
        self.place_piece(move.piece, move.x, move.y,move.pts_per_move)
        self.move_list.append(move)
    def undo_move(self):
        last_move = self.move_list.pop()
        self.next_pieces.insert(0,self.current_piece)
        self.current_piece = last_move.piece
        self.score =- last_move.pts_per_move
        self.un_place_piece(self.current_piece, last_move.x, last_move.y, last_move.pts_per_move)
    def un_place_piece(self,piece, x, y, pts_per_move):
        if self.can_place_piece(piece, x, y):
            shape = piece.get_shape()
            rows, cols = shape.shape
            
            for i in range(rows):
                for j in range(cols):
                    if shape[i, j] != 0: # Place non-zero parts of the piece
                        if shape[i, j] == 3:
                            if self.board[y + i, x + j] != 1:
                                self.board[y + i, x + j] = 1
                            else:
                                self.board[y+i, x+j] = 0

                        else:
                            if self.board[y + i, x + j] != 1:
                                self.board[y + i, x + j] = 1
                            else:
                                self.board[y+i, x+j] = 0
                            
                            
                            
            self.board[self.board  == 3] = 1
                        
             # Generate a new piece for the queue
            return True
        return False

        
        


    def generate_random_piece(self):
        """Generates a random piece from a list of defined pieces."""
        pieces = [TwoPieceHorz(), ThreePiece(), TwoPieceVert(), ThreePiecePlus1Vert(),
                  OneLeftFourDown(), OnePiece(), HingePieceLeft(), OneVertPlusThreePiece(),
                  FivePiece(), OneLeftMidPlusThreeVert(), HingePieceRight(), ThreeHorzPlusTwoHorzDown()]
        return random.choice(pieces)

    def can_place_piece(self, piece, x, y):
        """Checks if a piece can be placed on the board at coordinates (x, y)."""
        shape = piece.shape
        piece_height, piece_width = shape.shape
        board_height, board_width = self.board.shape
        if x < 0 or y < 0 or x + piece_width > board_width or y + piece_height > board_height:
            return False  # Piece is out of bounds
        
        center_coords = np.argwhere(shape == 3)  # Find center of the piece
        if center_coords.size == 0:
            return False  # No center found in the piece
        
        center_y, center_x = center_coords[0]
        target_color = self.board[y + center_y, x + center_x]
        
        for i in range(piece_height):
            for j in range(piece_width):
                if shape[i, j] != 0:  # Non-zero values indicate piece cells
                    if self.board[y + i, x + j] != target_color:  # Check for color mismatch
                        return False

        return True # Piece can be placed
    def generate_legal_moves(self,board, current_piece, held_piece, next_pieces):
        board = np.copy(board)
        legal_moves = np.array([])
        for x in range(5):
            for y in range(5):
                if self.can_place_piece(current_piece, x, y):
                    move = Move(current_piece,x,y, hold = False)
                    legal_moves = np.append(legal_moves, (move))
                self.hold_piece()
                if self.can_place_piece(current_piece, x, y):
                    move = Move(current_piece,x,y, hold= True)
                    legal_moves = np.append(legal_moves, (move))
                self.unhold_piece()
        return legal_moves
    
    def place_piece(self, piece, x, y, pts_per_move):
        """Places the piece on the board if allowed."""
        if self.can_place_piece(piece, x, y):
            shape = piece.get_shape()
            rows, cols = shape.shape
            
            for i in range(rows):
                for j in range(cols):
                    if shape[i, j] != 0: # Place non-zero parts of the piece
                        if shape[i, j] == 3:
                            if self.board[y + i, x + j] != 1:
                                self.board[y + i, x + j] = 1
                            else:
                                self.board[y+i, x+j] = 0

                        else:
                            if self.board[y + i, x + j] != 1:
                                self.board[y + i, x + j] = 1
                            else:
                                self.board[y+i, x+j] = 0
                            
                            
                            
            self.board[self.board  == 3] = 1
                        
            self.score += pts_per_move  # Increase score on valid placement
            self.current_piece = self.next_pieces.pop(0)  # Get the next piece
            self.next_pieces.append(self.generate_random_piece())  # Generate a new piece for the queue
            return True
        return False
    def check_if_board_cleared(self):
        if len(np.unique(self.board)) == 1 and self.score!= 0:
            self.pts_per_move += 1

    def display_board(self):
        """Displays the current board on the console."""
        print(self.board)

    def reset(self):
        """Resets the game to its initial state."""
        self.board = np.zeros((5, 5))  # Empty board
        self.pts_per_move = 1
        self.score = 0
        self.current_piece = None
        self.held_piece = None
        self.next_pieces = []
class Move():
    def __init__(self, piece, x,y,pts_per_move = 0,hold= False):
        self.piece = piece
        self.x = x
        self.y = y
        self.hold = hold
        self.pts_per_move = pts_per_move
        
class Piece:
    def __init__(self, shape):
        """Piece class initializes with a shape (2D numpy array)."""
        self.shape = np.array(shape)

    def rotate(self):
        """Rotates the piece 90 degrees clockwise."""
        self.shape = np.rot90(self.shape, k=-1)

    def get_shape(self):
        """Returns the shape of the piece."""
        return self.shape

    def get_name(self):
        """Returns the class name of the piece."""
        return self.__class__.__name__

    def __repr__(self):
        """String representation of the piece."""
        return f"{self.get_name()}(shape=\n{self.shape})"

# Define individual pieces as subclasses of the Piece class
class TwoPieceHorz(Piece):
    def __init__(self):
        shape = [[1, 3]]  # Horizontal piece
        super().__init__(shape)

    def rotate(self):
        """Rotation switches horizontal to vertical."""
        self.shape = np.rot90(self.shape, k=-1)

class ThreePiece(Piece):
    def __init__(self):
        shape = [[1, 1, 3]]  # Horizontal piece with three blocks
        super().__init__(shape)

class TwoPieceVert(Piece):
    def __init__(self):
        shape = [[1], [3]]  # Vertical piece
        super().__init__(shape)

class ThreePiecePlus1Vert(Piece):
    def __init__(self):
        shape = [[0, 0, 1], [1, 1, 3]]  # L-shaped piece
        super().__init__(shape)

class OneLeftFourDown(Piece):
    def __init__(self):
        shape = [[0, 1], [0, 1], [0, 1], [1, 3]]  # Vertical piece with a single block on the side
        super().__init__(shape)

class OnePiece(Piece):
    def __init__(self):
        shape = [[3]]  # Single block piece
        super().__init__(shape)

class HingePieceLeft(Piece):
    def __init__(self):
        shape = [[0, 1], [1, 3]]  # L-shaped hinge piece
        super().__init__(shape)

class OneVertPlusThreePiece(Piece):
    def __init__(self):
        shape = [[1, 0, 0], [1, 1, 3]]  # T-shaped piece
        super().__init__(shape)

class FivePiece(Piece):
    def __init__(self):
        shape = [[1, 1, 1, 1, 3]]  # Long horizontal piece
        super().__init__(shape)

class OneLeftMidPlusThreeVert(Piece):
    def __init__(self):
        shape = [[0, 1], 
                 [1, 1], 
                 [0, 3]]  # Vertical T-shaped piece
        super().__init__(shape)

class HingePieceRight(Piece):
    def __init__(self):
        shape = [[0, 1], [1, 3]]  # Mirror of HingePieceLeft
        super().__init__(shape)

class ThreeHorzPlusTwoHorzDown(Piece):
    def __init__(self):
        shape = [[1, 1, 1, 0], [0, 0, 1, 3]]  # Complex L-shaped piece
        super().__init__(shape)

def get_known_pieces():
    """Returns a dictionary of all known pieces with their shapes."""
    piece_classes = [
        TwoPieceHorz, ThreePiece, TwoPieceVert, ThreePiecePlus1Vert,
        OneLeftFourDown, OnePiece, HingePieceLeft, OneVertPlusThreePiece,
        FivePiece, OneLeftMidPlusThreeVert, HingePieceRight, ThreeHorzPlusTwoHorzDown
    ]
    known_pieces = {}
    for cls in piece_classes:
        piece_instance = cls()  # Create an instance of the piece
        piece_name = piece_instance.get_name()  # Get the piece's name
        known_pieces[piece_name] = piece_instance.get_shape()  # Store the shape
    return known_pieces

# Function to handle mouse click events on the game grid
def handle_click(game, pos):
    x, y = pos
    grid_x = x // CELL_SIZE
    grid_y = y // CELL_SIZE
    if grid_x < BOARD_SIZE and grid_y < BOARD_SIZE:
        move = Move(game.current_piece, grid_x, grid_y, game.pts_per_move)
        if game.can_place_piece(game.current_piece, grid_x, grid_y):
            game.play_move(move)

# Function to display next pieces info on the screen
def display_next_piece_info(game):
    next_piece = game.next_pieces[0]
    piece_name = next_piece.get_name()
    piece_shape = next_piece.get_shape()
    shape_str = '\n'.join([''.join(['#' if cell else '.' for cell in row]) for row in piece_shape])
    text = f"Next Piece: {piece_name}\nShape:\n{shape_str}"
    print(next_piece, piece_name, piece_shape, text)
    print("next pieces")
    for piece in game.next_pieces:
        print(f"{piece.get_name()}")
    print(game.next_pieces)
def display_current_piece_info(game):
    current_piece = game.current_piece
    piece_name = current_piece.get_name()
    piece_shape = current_piece.get_shape()
    shape_str = '\n'.join([''.join(['#' if cell else '.' for cell in row]) for row in piece_shape])
    text = f"current Piece: {piece_name}\nShape:\n{shape_str}"
    print(current_piece, piece_name, piece_shape, text)
# Function to draw the game board
def draw_board(board):
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            rect = pygame.Rect((x) * CELL_SIZE, (y) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if board[y, x] == 0 else BLUE
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

# Function to draw individual pieces on the board
def draw_piece(piece, top_left_x, top_left_y, cell_size):
    shape = piece.get_shape()
    for y in range(shape.shape[0]):
        for x in range(shape.shape[1]):
            if shape[y, x] != 0:
                rect = pygame.Rect(
                    top_left_x + x * cell_size,
                    top_left_y + y * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)  # Cell border

# Function to draw the next pieces in the queue
def draw_next_pieces(pieces):
    x_offset = BOARD_WIDTH + MARGIN
    y_offset = 0
    try:
        for piece in pieces:
            draw_piece(piece, x_offset, y_offset)
            y_offset += (piece.get_shape().shape[0] + 1) * CELL_SIZE
    except TypeError:
        if not isinstance(pieces, FivePiece):
            draw_piece(pieces, x_offset, y_offset, cell_size= CELL_SIZE)
            y_offset += (pieces.get_shape().shape[0] + 1) * CELL_SIZE
        elif isinstance(pieces, FivePiece):
            draw_piece(pieces, x_offset, y_offset, cell_size= CELL_SIZE/2)
            y_offset += (pieces.get_shape().shape[0] + 1) 

          # Move down for the next piece


# Main game loop
def main():
    game = Game()  # Initialize the game
    running = True
    clock = pygame.time.Clock()  # Used to control the frame rate
    show_next_piece_info = False  # Flag to show next piece info
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(game, pygame.mouse.get_pos())  # Handle mouse click
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_next_piece_info = True
                if event.key == pygame.K_r:
                    if game.held_piece == None:
                        game.hold_piece()
                    else:
                        game.unhold_piece()
                if event.key == pygame.K_g:
                    legal_moves = game.generate_legal_moves(game.board, game.current_piece, game.held_piece, game.next_pieces)
                    for x in legal_moves:
                        print(x.piece, x.y, x.x, x.hold)
                        print(x)
                        print(...)
                    print(len(legal_moves))
                if event.key == pygame.K_u:
                    game.undo_move()

          # Show next piece info when space is pressed

        # Draw everything on the screen
        screen.fill(BLACK)  # Clear the screen with black
        draw_board(game.board)  # Draw the game board
        draw_next_pieces(game.current_piece)
        game.check_if_board_cleared()
         # Draw the next pieces queue
        if show_next_piece_info:
            game.display_board() 
            display_next_piece_info(game)
            display_current_piece_info(game)
            print(f"The points:{game.score}")
            print(f"pts per move {game.pts_per_move}")
            print("."*10)
              # Display next piece info
            show_next_piece_info = False  # Reset flag after displaying info
        pygame.display.flip()  # Update the screen
        clock.tick(30)  # Limit to 30 frames per second

    pygame.quit()  # Quit the game when the loop ends

# Run the game
if __name__ == "__main__":
    main()




state_size = 25+ 12*3         # Example: Flattened 5x5 board
num_actions = 600       # Fixed maximum moves
hidden_size = 128       # Size of hidden layers
lr = 0.001              # Learning rate
gamma = 0.99            # Discount factor
batch_size = 32
replay_memory_size = 10000

class Neural_net():

    pass
class Target_net():

    pass

def train():
    pass






