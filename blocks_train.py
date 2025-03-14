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

# Main game loop


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_relu(Z):
    return Z > 0

def huber_loss(y_true, y_pred, delta=0.1):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)


def derivative_huber(error, delta=1.0):
    # error is A3 - target
    return np.where(np.abs(error) <= delta, error, delta * np.sign(error))



class Q_Network():
    def __init__(self, arch):
        self.W1 = np.random.randn(arch[1], arch[0]) * np.sqrt(2.0/arch[0])
        self.V_w1 = np.random.randn(arch[1], arch[0]) * np.sqrt(2.0/arch[0])
        self.b1 = np.zeros((arch[1], 1))
        self.V_b1 = np.zeros((arch[1], 1))
        self.W2 = np.random.randn(arch[2], arch[1]) * np.sqrt(2.0/arch[1])
        self.V_w2 = np.random.randn(arch[2], arch[1]) * np.sqrt(2.0/arch[1])

        self.b2 = np.zeros((arch[2], 1))
        self.W_b1 = np.zeros((arch[2], 1))
        
        # Dueling streams
        self.W_value = np.random.randn(1, arch[2]) * np.sqrt(2.0/arch[2])
        self.V_wv =  np.random.randn(1, arch[2]) * np.sqrt(2.0/arch[2]) # Value stream (1 output)
        self.b_value = np.zeros((1, 1))
        self.W_bv = np.zeros((1, 1))
        self.W_adv = np.random.randn(arch[3], arch[2]) * np.sqrt(2.0/arch[2])
        self.V_wa = np.random.randn(arch[3], arch[2]) * np.sqrt(2.0/arch[2])  # Advantage stream (N actions)
        self.b_adv = np.zeros((arch[3], 1))
        self.W_ba =  np.zeros((arch[3], 1))
    def forward(self, input):
        Z1 = np.dot(self.W1, input) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = relu(Z2)
        # Value stream (scalar per state)
        V = np.dot(self.W_value, A2) + self.b_value
        
        # Advantage stream (one per action)
        A = np.dot(self.W_adv, A2) + self.b_adv
        
        # Combine streams
        Q = V + (A - np.mean(A, axis=0, keepdims=True))  # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        
        return Z1, A1, Z2, A2, Q
    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W_value, self.b_value, self.W_adv, self.b_adv]
    def set_weights(self, x):
        self.W1 = x[0]
        self.b1 = x[1]
        self.W2 = x[2]
        self.b2 = x[3]
        self.W_value = x[4]
        self.b_value = x[5]
        self.W_adv = x[6]
        self.b_adv = x[7]

        
    def back_propogate(self, Z1, A1, Z2, A2, Q, state, target):
        m = state.shape[1]  # Batch size

        # Calculate Q error (Huber derivative)
        error = Q - target
        dQ = derivative_huber(error, delta=1.0)

        # Gradients for the value stream (V)
        dV = np.sum(dQ, axis=0, keepdims=True)
          # Sum over actions (axis=0) to get (1, m)

        # Gradients for the advantage stream (A)
        dA = dQ - np.mean(dQ, axis=0, keepdims=True)
         # (50, m)

        # Split gradients for value and advantage streams
        # Value stream gradients
        dW_value = (1/m) * np.dot(dV, A2.T)  # A2 (256, m) @ dV.T (m, 1) → (256, 1)
        db_value = (1/m) * np.sum(dV, axis=1, keepdims=True)  # Sum over batch (axis=1)

        # Advantage stream gradients
        dW_adv = (1/m) * np.dot(dA, A2.T)  # A2 (256, m) @ dA.T (m, 50) → (256, 50)
        db_adv = (1/m) * np.sum(dA, axis=1, keepdims=True)  # Sum over batch (axis=1)

        # Backpropagate through shared layers
        # Corrected: Use W_value and W_adv instead of their transposes
        dZ2_shared = (
            np.dot(self.W_value.T, dV) +  # (256, 1) @ (1, m) → (256, m)
            np.dot(self.W_adv.T, dA)      # (256, 50) @ (50, m) → (256, m)
        ) * derivative_relu(Z2)  # Z2 shape (256, m)

        dW2 = (1/m) * np.dot(dZ2_shared, A1.T)  # (256, m) @ (m, 256) → (256, 256)
        db2 = (1/m) * np.sum(dZ2_shared, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2_shared) * derivative_relu(Z1)  # W2 (256, 256) → (256, 256).T @ (256, m) → (256, m)
        dW1 = (1/m) * np.dot(dZ1, state.T)  # (256, m) @ (m, 86) → (256, 86)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2, dW_value, db_value, dW_adv, db_adv
    def weight_update(
    self, 
    dW1, db1, 
    dW2, db2, 
    dW_value, db_value, 
    dW_adv, db_adv, 
    alpha
):
    # Update shared layers
        self.W1 -= alpha * np.nan_to_num(dW1, nan=0.0)
        self.b1 -= alpha * np.nan_to_num(db1, nan=0.0)
        self.W2 -= alpha * np.nan_to_num(dW2, nan=0.0)
        self.b2 -= alpha * np.nan_to_num(db2, nan=0.0)

        # Update value stream
        self.W_value -= alpha * np.nan_to_num(dW_value, nan=0.0)
        self.b_value -= alpha * np.nan_to_num(db_value, nan=0.0)

        # Update advantage stream
        self.W_adv -= alpha * np.nan_to_num(dW_adv, nan=0.0)
        self.b_adv -= alpha * np.nan_to_num(db_adv, nan=0.0)
        


arch = [86, 256, 256, 50]
Q = Q_Network(arch)

def one_hot_held(held_piece, env):
    if held_piece is None:
        return np.zeros(12)
    else:
        vector = np.zeros(12)
        for x in range(12):
            if held_piece.__class__ == env.piece_types[x].__class__:
                vector[x] = 1
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


def learn(minibatch, batch_size, indices, gamma = 1, alpha = 0.1):
    #forward returns  A1, Z1, Z2, A2
    #we need  Z1, A1,Z2, A2, state, target
    #minibatch is composed of q_state, best_idx, reward, convert_state_to_vec(nxt_state,env), done 
    rewards = []
    states = []
    q_targets = []
    Z1_batch = []
    A1_batch = []
    Z2_batch = []
    A2_batch = []
    Q_batch = []
    
    states = np.hstack([sample[0][0] for sample in minibatch]) 
    for sample in minibatch:
        state = sample[0][0]
        q_state = state # Ensure q_state is always defined
        if sample[4]:
            # Terminal state: target is just the reward
            q_target = sample[2]

            Z1, A1, Z2, Q1 = Q_target.forward(q_state)
            Z1_batch.append(Z1)
            A1_batch.append(A1)
            Z2_batch.append(Z2)
            A2_batch.append(A2)
            Q_batch.append(Q1)
            
            
            # Create a target vector with the same shape as A2.
            # For terminal states, you might set all values to 0 and then set the chosen index to the reward.
            target_vector = copy.deepcopy(Q1)
            # Define best_idx appropriately; if there's no best action in terminal state,
            # you could choose an index (for instance, 0) or handle it differently.
            best_idx = np.argmax(target_vector)  # or some logic to choose a default
            target_vector[best_idx] = q_target
            q_targets.append(target_vector)
        else:
            best_idx = sample[1] 
            q_state = sample[0][0]
            next_state = sample[3]
            Z1, A1, Z2, A2, Q1 = Q_target.forward(q_state)
            Z1_batch.append(Z1)
            A1_batch.append(A1)
            Z2_batch.append(Z2)
            A2_batch.append(A2)
            Q_batch.append(Q1)
            if np.isnan(Q1).any():
                print("NaN detected in A2 before update")
                return
            if not (0 <= best_idx < len(Q1)):
                print(f"Invalid best_idx: {best_idx}")
            future_reward = max(Q_target.forward(next_state)[4])
            reward = sample[2]
            q_target = reward + gamma* future_reward
            q_append = copy.deepcopy(Q1)
            q_append[best_idx] = q_target
            q_targets.append(q_append)
    
    
    Z1 = np.hstack(Z1_batch)  # (60, batch_size)
    A1 = np.hstack(A1_batch)  # (60, batch_size)
    Z2 = np.hstack(Z2_batch)  # (50, batch_size)
    A2 = np.hstack(A2_batch)
    Q1 = np.hstack(Q_batch)  # (50, batch_size)  # (50, batch_size)
    targets = np.hstack(q_targets)
    targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
    
    loss = huber_loss(targets, Q1, 0.5)
    if random.random() < 0.01:
        print("loss: ",np.argmax(loss))
    dW1, db1, dW2, db2, dW_value, db_value, dW_adv, db_adv = Q.back_propogate(Z1, A1, Z2, A2, Q1, states, targets)

    
    
    Q.weight_update(dW1, db1, dW2, db2, dW_value, db_value, dW_adv, db_adv, alpha)
    
    new_td_errors = []
    update_priorities_in_deque(D, indices, new_td_errors)
    return np.argmax(loss)
    
def update_priorities_in_deque(deque, indices, new_td_errors):
    for x in range(len(indices)):
        idx = indices[x]
        (q_state, full_state_cache), best_idx, reward, next_state, done, td_error = deque[idx]
        new_td_error = new_td_errors[x]
        deque[idx] = ((q_state, full_state_cache), best_idx, reward, next_state, done, new_td_error)

    
    
    

        

    
         #add more here mainly about finding predicted vals of Q network for each state :)
    #find loss then preform gradient descent using loss

    

# Q'

Q_target = copy.deepcopy(Q) #Q' NeuralNetwork(same parms as above) then update_target(Q_target.NN) will also work

# Replay Memory
D = deque(maxlen=10000) # if D==maxlen and we append new data oldest one will get removed


# Epsilon
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Gamma
gamma = 0.95



def get_batch(D, batch_size):
    batch = random.sample(D, batch_size)
    states, actions, rewards, new_states, done = zip(*batch)
    return states, actions, rewards, new_states, done 
# Just to check the highest score obtained during training
best_score = -np.inf
    
def convert_vect_move_no_hold(q_values, legal_moves, env):
    legal_indices = []
    masked_q = np.full_like(q_values, -np.inf)
    for move in legal_moves:
        # Calculate index accounting for piece dimensions
        if move.x + move.piece.shape.shape[1] > 5: continue
        if move.y + move.piece.shape.shape[0] > 5: continue
        idx = move.x + move.y*5 
        legal_indices.append(idx)
    masked_q[legal_indices] = q_values[legal_indices]
    best_idx = np.argmax(masked_q)
    
    base_idx = best_idx
    x = base_idx % 5
    y = base_idx // 5
    
    
    for move in legal_moves:
        if move.x == x and move.y == y and move.hold == False:
            return move, best_idx
        
    

    return 0
    # Select best valid move
    
    


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
    return np.concatenate((state.flatten(), one_hot_held(env.current_piece, env), one_hot_held(env.held_piece,env), one_hot_next_pieces(env), [env.pts_per_move])).reshape(-1,1)




def sample(deque, batch_size, alpha = 0.6):
    #what is alpha? alpha determines the uniformity basically like how niormal the distribution is. NEED to tune
   
    priorities = [abs(td_error) + 1e-6 for _, _, _, _, _, td_error in deque] 
    probabilities = [n**alpha for n in priorities]
    
    prob_sum = np.sum(probabilities)  
    probabilities /= prob_sum
    probabilities = list(probabilities)
    
    # so we need the indices and the actual experieces. The indices are needed why? to uodate later
    indices = np.random.choice(len(deque),size = batch_size, p = probabilities) 
    experiences = [deque[idx] for idx in indices]
    return experiences, indices
def train(num_episode=1000,batch_size=128,C=1000, ep= 20, gamma = 1, tau = 0.005, alpha = 1 ):
    running = True
    global epsilon,best_score
    steps = 0
    env = Game()
    num_episodes = 100  # Moving average window
    episode_scores = [] 
    episodes_rewards = []
    episode_losses = []
    average_losses = []
    current_song = random.choice(play_lists)
    scores = []
    rewards = []
    lr = 2.5e-4
    for i in range(1,num_episode+1):
        episode_reward = 0
        episode_loss = 0
        episode_score = 0


        # Sample Phase
        done = False
        nxt_state,_ = env.reset()
        env.pts_per_move = 1
        q_state = None
        while not done:
            
            
            state = nxt_state
            epsilon = max(epsilon_min,epsilon*epsilon_decay) # e decay
            lr = max(lr * 0.999, 1e-5)

            # e-greedy(Q)
            if np.random.rand() < epsilon: 
                hold_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)[2]
                if len(hold_moves) == 0:
                    episode_score += env.score
                    
                    done = True
                    print("random")
                    break;
                action = random.choice(hold_moves)
                best_idx = (int(action.hold)+1)* (action.y*5 + action.x)
                full_state_cache = [state, env.current_piece, env.held_piece, env.next_pieces, env.pts_per_move]
                q_state = convert_state_to_vec(state, env)
            else:
                legal_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)[2]
                if len(legal_moves) == 0:
                    episode_score += env.score
                    done = True 
                    break; 
                full_state_cache = [state, env.current_piece, env.held_piece, env.next_pieces, env.pts_per_move]
                q_state = convert_state_to_vec(state, env)
                action = Q.forward(q_state)[4].flatten()
                action, best_idx = convert_vec_to_move(action, legal_moves, env)
                
            if env.score > best_score:
                best_score = env.score
            if env.pts_per_move != 1:
                print("board cleared or filled")
            nxt_state,reward,done, score = env.play_move(action)
            episode_reward += reward
            episode_score += score
            nxt_q_values = Q_target.forward(convert_state_to_vec(nxt_state, env))[4]  # Get Q-values
            td_error = int(abs(reward + gamma * np.max(nxt_q_values) - Q.forward(q_state)[4][best_idx]))
            D.append(((q_state, full_state_cache), best_idx, reward, convert_state_to_vec(nxt_state, env), done, td_error))
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
            # Learining Phase

            if len(D) >= batch_size:
                minibatch, indices = sample(D, batch_size= batch_size)
                loss = learn(minibatch, batch_size,indices, gamma = gamma,  alpha = lr)
                episode_losses.append(loss)
            pygame.display.flip()  # Update the screen
            clock.tick(1000000)
            
            steps+=1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            if steps%C ==0: update_target(Q_target)
        episode_scores.append(episode_score)
        episodes_rewards.append(episode_reward)
            
        if i % ep == 0:
            average_score = sum(episode_scores)/len(episode_scores)
            # Calculate the moving average of rewards over the last 'num_episodes' episodes
            if len(episode_scores) > num_episodes:
                episode_scores.pop(0)  # Keep the last 'num_episodes' rewards
                average_score = sum(episode_scores) / len(episode_scores)
            average_reward = sum(episodes_rewards)/len(episodes_rewards)
            # Calculate the moving average of rewards over the last 'num_episodes' episodes
            if len(episodes_rewards) > num_episodes:
                episodes_rewards.pop(0)  # Keep the last 'num_episodes' rewards
                average_reward = sum(episodes_rewards) / len(episodes_rewards)
            average_episode_loss = sum(episode_losses) / len(episode_losses)/ 128

            scores.append(average_score)
            rewards.append(average_reward)
            average_losses.append(average_episode_loss)
            
                
            
            print("\n" * 2)
            print(f"Episode: {i} Reward: {episode_reward} and best score: {best_score} and done: {done}")
            print("Average Reward:", average_reward)
            print("Average Score:", average_score)
            
        
    return Q.get_weights(), scores, rewards, average_losses




weights, scores, rewards, average_losses = train()









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
test = Game()

main(test)"""





import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")  # Use Agg backend (no GUI)

print(average_losses)
# Example reward array
 # X-axis: episode numbers
episodes = list(range(len(rewards)))
episodes = [n*20 for n in episodes]
# Create the plot
plt.figure(figsize=(20, 10))  # Set figure size
plt.plot(episodes, scores, marker='o', linestyle='-', color='b', label="scores")  # Plot with markers
plt.plot(episodes, rewards, marker='s', linestyle='--', color='r', label="Rewards") 
plt.plot(episodes, average_losses, marker='s', linestyle='--', color='r', label="loss")# Plot loss

# Labels and title
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Score per Episode")
plt.legend()
plt.grid(True)  # Add grid for better visualization

# Show the plot
plt.show()