import numpy as np
import random
import pygame
import numpy as np
from collections import deque
import random
from collections import deque
import copy
from copy import deepcopy
import time
import os

OBSIDIAN = (20, 18, 35)       # Backgrounds, large elements
AMETHYST = (88, 47, 161)    # Buttons, accents
GOLD = (193, 154, 107)       # Text, borders, fine details
WHITE = (237, 234, 229)
HUSHED_SKY = (155, 168, 183)


# Set window position (e.g., x=100, y=100 pixels from top-left corner)
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
# Initialize Pygame
pygame.init()
pygame.mixer.init()
MUSIC_END_EVENT = pygame.USEREVENT + 1
pygame.mixer.music.set_endevent(MUSIC_END_EVENT)

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
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pygame Board Game with Next Pieces")
font = pygame.font.SysFont(None, 24)

# Define colors

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
        self.piece_types = pieces = [TwoPieceHorz(), ThreePiece(), TwoPieceVert(), ThreePiecePlus1Vert(),
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
        if move.hold == True:
            self.hold_piece()
        self.place_piece(move.piece, move.x, move.y,move.pts_per_move)
        self.move_list.append(move)
        self.pts_per_move_list.append(self.pts_per_move)
        self.check_if_board_cleared()

        reward = self.pts_per_move*10 + 0 if len(self.move_list) < 0 else abs(np.count_nonzero(self.board == 1 ) - abs(np.count_nonzero(self.board == 0)))
        print(f" reward: {reward}")
        return self.board, reward, self.done
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
        if move in self.generate_legal_moves(self.board, self.current_piece, self.held_piece, self.next_pieces):
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
            test_game = deepcopy(self)  # Use current instance's state
            legal_moves = []
            
            # Check moves without holding
            for x in range(5):
                for y in range(5):
                    if self.can_place_piece(current_piece, x, y):
                        legal_moves.append(Move(current_piece, x, y, hold=False))
            
            # Check moves with holding (swap current and held pieces)
            if held_piece is None:
                # Simulate holding current piece and taking next piece
                test_game.hold_piece()
                new_current = test_game.current_piece
                for x in range(5):
                    for y in range(5):
                        if test_game.can_place_piece(new_current, x, y):
                            legal_moves.append(Move(new_current, x, y, hold=True))
            else:
                # If already holding, simulate swapping
                test_game.unhold_piece()
                for x in range(5):
                    for y in range(5):
                        if test_game.can_place_piece(test_game.current_piece, x, y):
                            legal_moves.append(Move(test_game.current_piece, x, y, hold=True))
            
            if not legal_moves:
                self.done = True
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
            color = WHITE if board[y, x] == 0 else AMETHYST
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (59,59,88), rect, 1)  # Cell border

# Function to draw individual pieces on the board
def draw_piece(piece, top_left_x, top_left_y, cell_size, color = BLUE):
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
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (74, 74, 74), rect, 1)  # Cell border

# Function to draw the next pieces in the queue

def draw_current_piece_label():
    text_surface = font.render(f"Current Piece:", True, GOLD)
    x_offset = BOARD_WIDTH + MARGIN+ 25
    y_offset = 25
    screen.blit(text_surface, (x_offset, y_offset))
def draw_current(pieces):
    x_offset = 25 + BOARD_WIDTH + MARGIN
    y_offset = 50
    try:
        for piece in pieces:
            draw_piece(piece, x_offset, y_offset)
            y_offset += (piece.get_shape().shape[0] + 1) * CELL_SIZE
    except TypeError:
        if not isinstance(pieces, FivePiece):
            draw_piece(pieces, x_offset, y_offset, cell_size= CELL_SIZE//2, color = (138, 163, 155))
            y_offset += (pieces.get_shape().shape[0] + 1) * CELL_SIZE//2
        elif isinstance(pieces, FivePiece):
            draw_piece(pieces, x_offset, y_offset, cell_size= CELL_SIZE//2, color = (138, 163, 155))
            y_offset += (pieces.get_shape().shape[0] + 1) * CELL_SIZE//2

          # Move down for the next piece
def draw_held_piece_label():
    text_surface = font.render(f"Held Piece:", True, GOLD)
    x_offset = BOARD_WIDTH + MARGIN+ 300
    y_offset = 300
    screen.blit(text_surface, (x_offset, y_offset))







def draw_held_piece(piece):
    if piece is None:
        return 0
    x_offset = 300 + BOARD_WIDTH + MARGIN
    y_offset = 350
    
    if not isinstance(piece, FivePiece):
        draw_piece(piece, x_offset, y_offset, cell_size= CELL_SIZE//2, color = HUSHED_SKY)
        y_offset += (piece.get_shape().shape[0] + 1) * CELL_SIZE//2
    elif isinstance(piece, FivePiece):
        draw_piece(piece, x_offset, y_offset, cell_size= CELL_SIZE/2, color = HUSHED_SKY)
        y_offset += (piece.get_shape().shape[0] + 1) //2
def draw_score(score):
    text_surface = font.render(f"Score: {score}", True, GOLD)
    x_offset = BOARD_WIDTH + MARGIN+ 300
    y_offset = 25
    screen.blit(text_surface, (x_offset, y_offset))
def draw_pts_per_move(pts_per_move):
    text_surface = font.render(f"Points per move: {pts_per_move}", True, GOLD)
    x_offset = BOARD_WIDTH + MARGIN+ 300
    y_offset = 100
    screen.blit(text_surface, (x_offset, y_offset))

def draw_next_piece_label():
    text_surface = font.render(f"Next Piece:", True, GOLD)
    x_offset = BOARD_WIDTH + MARGIN+ 25
    y_offset = 300
    screen.blit(text_surface, (x_offset, y_offset))
def draw_next_piece(piece):
    if piece is None:
        return 0
    x_offset = 25 + BOARD_WIDTH + MARGIN
    y_offset = 350
    
    if not isinstance(piece, FivePiece):
        draw_piece(piece, x_offset, y_offset, cell_size= CELL_SIZE//2, color = (199, 168, 163))
        y_offset += (piece.get_shape().shape[0] + 1) * CELL_SIZE//2
    elif isinstance(piece, FivePiece):
        draw_piece(piece, x_offset, y_offset, cell_size= CELL_SIZE/2, color = (199, 168, 163))
        y_offset += (piece.get_shape().shape[0] + 1) //2

# Main game loop



def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))




class Q_Network():
    def __init__(self,arch):
        self.W1 = np.random.randn(arch[1], arch[0]) * 0.01
        self.b1 = np.zeros((arch[1], 1))
        self.W2 = np.random.randn(arch[2], arch[1]) * 0.01
        self.b2 = np.zeros((arch[2],1))
    def forward(self, input):
        A1 = np.dot(self.W1, input) + self.b1
        Z1 = relu(A1)
        Z2 = np.dot(self.W2, Z1) + self.b2
        return Z2
    def get_weights(self):
        return self.W1, self.b1, self.W2, self.b2
    def set_weights(self, x):
        self.W1 = x[0]
        self.b1 = x[1]
        self.W2 = x[2]
        self.b2 = x[3]
    def back_propogate(self):
        pass


arch = [74, 60, 50]
Q = Q_Network(arch)

def one_hot_held(env):
    if env.held_piece is None:
        return np.zeros(12)
    else:
        vector = np.zeros(12)
        for x in range(12):
            if env.held_piece.__class__ == env.piece_types[x].__class__:
                vector[x] = 1
    print(vector)
    return vector

def one_hot_next_pieces(env):
    vector_1 = np.zeros(12)
    vector_2 = np.zeros(12)
    vector_3 = np.zeros(12)
    vectors = [vector_1, vector_2, vector_3]
    for x in range(3):
        for y in range(12):
            if env.next_pieces[x].__class__ == env.piece_types[y].__class__:
                vectors[x][y] = 1
    result = np.concatenate((vector_1, vector_2, vector_3))
    return result

def update_target(target):
    target.set_weights(Q.get_weights())
    return 0
    



def learn(minibatch, batch_size):
    states, actions, rewards, next_states, dones = zip(*minibatch)
    target_q_vals = []
    for sample in minibatch:
        if minibatch[4]:
            target_Q_val = minibatch[2]
        else:
            next_state_Q_values = Q_target.forward(minibatch[3])  
            max_next_Q = max(next_state_Q_values)  
            target_Q_val = minibatch[2] + gamma * max_next_Q
        target_q_vals.append(target_Q_val)
        predicted_vals = []
    for sample in minibatch:
        for state, action, reward, next_state, done in sample:
            predicted_val = Q.forward(state)
        predicted_vals.append(predicted_val)
         #add more here mainly about finding predicted vals of Q network for each state :)
    #find loss then preform gradient descent using loss

    

# Q'

Q_target = copy.deepcopy(Q) #Q' NeuralNetwork(same parms as above) then update_target(Q_target.NN) will also work

# Replay Memory
D = deque(maxlen=10000) # if D==maxlen and we append new data oldest one will get removed


# Epsilon
epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995

# Gamma
gamma = 0.95



def get_batch(D, batch_size):
    batch = random.sample(D, batch_size)
    states, actions, rewards, new_states, done = zip(*batch)
    return states, actions, rewards, new_states, done 
# Just to check the highest score obtained during training
best_score = -np.inf
    


def convert_vec_to_move(q_values, legal_moves, env):
    # Initialize all actions as invalid
    masked_q = np.full_like(q_values, -np.inf)
    
    # Create legal move indices
    legal_indices = []
    for move in legal_moves:
        # Calculate index accounting for piece dimensions
        if move.x + move.piece.shape.shape[1] > 5: continue
        if move.y + move.piece.shape.shape[0] > 5: continue
        idx = move.x + move.y*5 + (25 if move.hold else 0)
        legal_indices.append(idx)
    
    # Apply legal moves to mask
    masked_q[legal_indices] = q_values[legal_indices]
    
    # Select best valid move
    best_idx = np.argmax(masked_q)
    
    # Decode index to move parameters
    hold = best_idx >= 25
    base_idx = best_idx - 25 if hold else best_idx
    x = base_idx % 5
    y = base_idx // 5
    
    # Verify against legal moves
    for move in legal_moves:
        if move.x == x and move.y == y and move.hold == hold:
            return move, best_idx
    
    # Fallback to random legal move
    return np.random.choice(legal_moves)


def convert_state_to_vec(state, env):
    return np.concatenate((state.flatten(),one_hot_held(env), one_hot_next_pieces(env), [env.pts_per_move])).reshape(-1,1)




def train(num_episode=100,batch_size=32,C=10,ep=10):
    running = True
    clock = pygame.time.Clock()
    global epsilon,best_score
    steps = 0
    env = Game()
    
    for i in range(1,num_episode+1):
        episode_reward = 0
        episode_loss = 0


        # Sample Phase
        done = False
        nxt_state,_ = env.reset()
        env.pts_per_move = 1
        while not done:
            screen.fill(OBSIDIAN)  # Clear the screen with black
            draw_current_piece_label()
            draw_next_piece_label()
            draw_board(env.board)  # Draw the game board
            draw_score(env.score)
            draw_current(env.current_piece)
            draw_next_piece(env.next_pieces[0])
            draw_held_piece_label()
            draw_held_piece(env.held_piece)
            env.check_if_board_cleared()
            try:
                draw_held_piece(env.held_piece)
            except AttributeError:
                pass
            pygame.display.flip()  # Update the screen
            clock.tick(30)
            state = nxt_state
            q_state = None
            epsilon = max(epsilon_min,epsilon*epsilon_decay) # e decay

            # e-greedy(Q)
            if np.random.rand() < epsilon: 
                legal_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)
                if len(legal_moves) == 0:
                    done = True
                    break;
                   
                action = np.random.choice(legal_moves)
                best_idx = (int(action.hold)+1)* (action.y*5 + action.x)
                print("random move")
            else:
                print("next move:")
                legal_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)
                if len(legal_moves) == 0:
                    done = True
                    break;
                    
                q_state = convert_state_to_vec(state, env)
                action = Q.forward(q_state).flatten()
                action, best_idx = convert_vec_to_move(action, legal_moves,env)
                print("coordinates of placement",action.x+1, action.y+1)
                print("\n")
                if action.hold == True:
                    print("The held piece")
                    print(env.held_piece if env.held_piece is not None else env.next_pieces[0])
                    print("\n")
                print(action.return_params())
                print("\n")
                print(env.score)

            nxt_state,reward,done = env.play_move(action)
            episode_reward += reward
            D.append((q_state, best_idx, reward, convert_state_to_vec(nxt_state,env), done))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
            # Learining Phase

            if len(D) >= batch_size:
                minibatch = random.sample(D, batch_size)
                learn(minibatch, batch_size)
            steps+=1
            
            if steps%C ==0: update_target(Q_target)
        if episode_reward > best_score:
            best_score = episode_reward

        if i%ep==0: 
            print("\n"*2)
            print(f"Episode: {i} Reward: {episode_reward} and best score: {best_score} and {done}")




train()















"""import time
def main(env):
    play = True
    game = env  # Initialize the game
    running = True
    clock = pygame.time.Clock()  # Used to control the frame rate
    show_next_piece_info = False
    current_song = random.choice(play_lists)
    pygame.mixer.music.load(current_song)
    pygame.mixer.music.play()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == MUSIC_END_EVENT:
                if play == True:
                    next_song = current_song
                    while next_song == current_song:  # Ensure new song is different
                        next_song = random.choice(play_lists)
                    current_song = next_song
                    pygame.mixer.music.load(current_song)
                    pygame.mixer.music.play()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(game, pygame.mouse.get_pos())  # Handle mouse click
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if current_song is not None:
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        play = False
                        current_song = None
                    else:
                        play = True
                        pygame.mixer.music.stop()
                        pygame.mixer.music.unload()
                        next_song = current_song
                        while next_song == current_song:  # Ensure new song is different
                            next_song = random.choice(play_lists)
                        current_song = next_song
                        pygame.mixer.music.load(current_song)
                        pygame.mixer.music.play()

                if event.key == pygame.K_SPACE:
                    show_next_piece_info = True
                if event.key == pygame.K_h:
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
            screen.fill(OBSIDIAN)  # Clear the screen with black
            draw_current_piece_label()
            draw_next_piece_label()
            draw_board(env.board)  # Draw the game board
            draw_score(env.score)
            draw_current(env.current_piece)
            draw_next_piece(env.next_pieces[0])
            draw_held_piece_label()
            draw_held_piece(env.held_piece)
            draw_pts_per_move(env.pts_per_move)
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

    pygame.quit()  # Quit the game when the loop end
"""