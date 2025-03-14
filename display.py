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
from blocks_env import *
from Q_network import *
from activations import *
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
