import numpy as np
import random
import pygame
import numpy as np
from collections import deque
import random
from collections import deque
import copy
from copy import deepcopy
import os

np.random.seed(421)
random.seed(421)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

OBSIDIAN = (20, 18, 35)       # Backgrounds, large elements
AMETHYST = (88, 47, 161)    # Buttons, accents
GOLD = (193, 154, 107)       # Text, borders, fine details
WHITE = (237, 234, 229)
HUSHED_SKY = (155, 168, 183)


# Set window position (e.g., x=100, y=100 pixels from top-left corner)
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
# Initialize Pygame


# Start with a random song


play_lists = [r"C:\Users\Sreek\Downloads\Auto_typer\storm-clouds-purpple-cat(chosic.com).mp3",r"C:\Users\Sreek\Downloads\Auto_typer\Sonder(chosic.com).mp3",r"C:\Users\Sreek\Downloads\Auto_typer\Ghostrifter-Official-Purple-Dream(chosic.com).mp3", r"C:\Users\Sreek\Downloads\Auto_typer\Heart-Of-The-Ocean(chosic.com).mp3" ]
# Constants for the game

# Play indefinitely (loop)
CELL_SIZE = 100  # Size of each cell in pixels
BOARD_SIZE = 5   # 5x5 board
NEXT_PIECES_COUNT = 3  # Number of next pieces to display
MARGIN = 20  # Margin between board and next pieces display


# Calculate window dimensions
BOARD_WIDTH = 100 * BOARD_SIZE
NEXT_PIECES_WIDTH = CELL_SIZE * 6 # Width allocated for next pieces
WINDOW_WIDTH = BOARD_WIDTH + MARGIN + NEXT_PIECES_WIDTH
WINDOW_HEIGHT = 100 * BOARD_SIZE+ 200

# Set up display
pygame.init()

# Game setup
screen = pygame.display.set_mode((1000, 800))  # Screen dimensions
pygame.display.set_caption('Your Game')  # Set game window title
clock = pygame.time.Clock()
MUSIC_END_EVENT = pygame.USEREVENT + 1  # Custom event for music end
pygame.mixer.music.set_endevent(MUSIC_END_EVENT)
# Define colors

font = pygame.font.SysFont(None, 24)

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
        next_piece = None
        self.next_pieces = [self.generate_random_piece()]
        for _ in range(11):
            while self.next_pieces[-1] == next_piece:
                next_piece = self.generate_random_piece()
            self.next_pieces.append(next_piece)
        self.next_pieces = [self.generate_random_piece() for _ in range(11)]
        self.move_list = []
        self.pts_per_move_list = [1]
        self.done = False
        self.piece_types = [TwoPieceHorz(), ThreePiece(), TwoPieceVert(), ThreePiecePlus1Vert(),
                  OneLeftFourDown(), OnePiece(), HingePieceLeft(), OneVertPlusThreePiece(),
                  FivePiece(), OneLeftMidPlusThreeVert(), HingePieceRight(), ThreeHorzPlusTwoHorzDown()]
    def hold_piece(self):
        if self.held_piece is None:
            self.held_piece =   self.current_piece
            self.current_piece = self.next_pieces.pop(0)  # Get the next piece
            self.next_pieces.append(self.generate_random_piece())
        else:
            self.unhold_piece()
    def unhold_piece(self):
        self.next_pieces.insert(0,self.current_piece)
        self.current_piece = self.held_piece
        self.held_piece = None

    def play_move(self, move):
        if move.hold:
            self.hold_piece()
        self.place_piece(move.piece, move.x, move.y, move.pts_per_move)
        self.move_list.append(move)
        self.pts_per_move_list.append(self.pts_per_move)
        
        num_zeros = (len(self.board)**2 - np.sum(self.board))
        closness_clear = abs(np.sum(self.board) - num_zeros)
        # Compute the raw reward based on your criteria. 
        reward = self.pts_per_move + closness_clear  # Base reward for valid move
        if self.check_if_board_cleared():
            reward += 10.0  # Bonus for clearing
        normalized_reward = 1 / (1 + np.exp(-reward))
        
        
        
        
        return self.board, normalized_reward, self.done, self.pts_per_move
    def undo_move(self):
        try:
            last_move = self.move_list.pop()
        except IndexError:
            print("Cannot undo Move: This is the first move")
        self.next_pieces.insert(0,self.current_piece)
        self.current_piece = last_move.piece
        self.pts_per_move_list.pop()
        try: 
            self.pts_per_move = self.pts_per_move_list[-1]
        except IndexError:
            self.pts_per_move = 1
        self.score -= last_move.pts_per_move
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
    

        
        
    def move_is_legal(self,move):
        if move in self.generate_legal_moves(self.board, self.current_piece, self.held_piece, self.next_pieces)[2]:
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
            test_game = deepcopy(self)  
            normal_moves = []
            hold_moves = []# Use current instance's state
            legal_moves = []
            
            # Check moves without holding
            for x in range(5):
                for y in range(5):
                    if self.can_place_piece(current_piece, x, y):
                        normal_moves.append(Move(current_piece, x, y, hold=False))
            
            # Check moves with holding (swap current and held pieces)
            if held_piece is None:
                # Simulate holding current piece and taking next piece
                test_game.hold_piece()
                new_current = test_game.current_piece
                for x in range(5):
                    for y in range(5):
                        if test_game.can_place_piece(new_current, x, y):
                            hold_moves.append(Move(new_current, x, y, hold=True))
            else:
                # If already holding, simulate swapping
                test_game.unhold_piece()
                for x in range(5):
                    for y in range(5):
                        if test_game.can_place_piece(test_game.current_piece, x, y):
                            hold_moves.append(Move(test_game.current_piece, x, y, hold=True))
            legal_moves  = normal_moves + hold_moves
            if not legal_moves:
                self.done = True
            return normal_moves, hold_moves, legal_moves
            
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
            return True
        return False

    def display_board(self):
        """Displays the current board on the console."""
        print(self.board)

    def reset(self):
        """Resets the game to its initial state."""
        self.board = np.zeros((5, 5))  # Empty 5x5 board
        self.pts_per_move = 1  # Points per valid move
        self.score = 0  # Current score
        self.held_piece = None  # No piece held initially
        self.current_piece = self.generate_random_piece()  # Generate the first piece
        self.next_pieces = [self.generate_random_piece() for _ in range(11)]
        self.move_list = []
        self.pts_per_move_list = []
        self.done = False
        self.piece_types = pieces = [TwoPieceHorz(), ThreePiece(), TwoPieceVert(), ThreePiecePlus1Vert(),
                  OneLeftFourDown(), OnePiece(), HingePieceLeft(), OneVertPlusThreePiece(),
                  FivePiece(), OneLeftMidPlusThreeVert(), HingePieceRight(), ThreeHorzPlusTwoHorzDown()]
        return self.board, self.current_piece
class Move():
    def __init__(self, piece, x,y,pts_per_move = 1,hold= False):
        self.piece = piece
        self.x = x
        self.y = y
        self.hold = hold
        self.pts_per_move = pts_per_move
    def return_params(self):
        print(self.piece, self.x, self.y, self.hold, self.pts_per_move)
    def __eq__(self, other):
        return (self.x == other.x and 
                self.y == other.y and 
                self.hold == other.hold and 
                self.piece.__class__ == other.piece.__class__) 
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
