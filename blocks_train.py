import numpy as np
import random
from collections import deque
import random
import copy
from copy import deepcopy
from blocks_env import *
from Q_network import *
from activations import *
from tqdm import tqdm
import time
np.random.seed(421)
random.seed(421)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


# Main game loop
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
        full_state_cache = sample[0][1]
        q_state = state # Ensure q_state is always defined
        if sample[4]:
            # Terminal state: target is just the reward
            q_target = sample[2]

            Z1, A1, Z2, A2, Q1 = Q_target.forward(q_state)
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
            q_idx = convert_vec_to_move(Q.forward(next_state)[4], full_state_cache[5][2])[1]
            future_reward = Q_target.forward(next_state)[4][q_idx]
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
    dW1, db1, dW2, db2, dW_value, db_value, dW_adv, db_adv = Q.back_propogate(Z1, A1, Z2, A2, Q1, states, targets)

    
    
    Q.weight_update(dW1, db1, dW2, db2, dW_value, db_value, dW_adv, db_adv, alpha)
    new_Q_batch = []
    for sample in minibatch:
        state = sample[0][0]
        q_state = state
        Z1, A1, Z2, A2, Q1 = Q_target.forward(q_state)
        new_Q_batch.append(np.max(Q1))
    td_targets = np.max(targets, axis = 0)

    new_td_errors = td_targets - new_Q_batch
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
D = deque(maxlen=100000) # if D==maxlen and we append new data oldest one will get removed


# Epsilon




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
    
    


def convert_vec_to_move(q_values, legal_moves):
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


epsilon_min = 0.00001

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
def train(num_episode=10000,batch_size= 32,C=1000, ep = 300, gamma = 1, tau = 0.005, alpha = 1, alpha_decay = 1, epsilon = 0.98 ):
    running = True
    best_score = -np.inf
    epsilon = epsilon
    steps = 0
    env = Game()
    num_episodes = 100  # Moving average window
    episode_scores = [] 
    episodes_rewards = []
    episode_losses = []
    average_losses = []
    scores = []
    rewards = []
    lr = alpha
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
            lr = max(lr * alpha_decay, 1e-5)

            # e-greedy(Q)
            if np.random.rand() < epsilon: 
                hold_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)[2]
                if len(hold_moves) == 0:
                    episode_score += env.score
                    
                    done = True
                    break;
                action = random.choice(hold_moves)
                best_idx = (int(action.hold)+1)* (action.y*5 + action.x)
                full_state_cache = [state, env.current_piece, env.held_piece, env.next_pieces, env.pts_per_move, env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)]
                q_state = convert_state_to_vec(state, env)
            else:
                legal_moves = env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)[2]
                if len(legal_moves) == 0:
                    episode_score += env.score
                    done = True 
                    break; 
                full_state_cache = [state, env.current_piece, env.held_piece, env.next_pieces, env.pts_per_move,env.generate_legal_moves(env.board, env.current_piece, env.held_piece, env.next_pieces)]
                q_state = convert_state_to_vec(state, env)
                action = Q.forward(q_state)[4].flatten()
                action, best_idx = convert_vec_to_move(action, legal_moves)
                
            if env.score > best_score:
                best_score = env.score
            if env.pts_per_move != 1:
                print("board cleared or filled")
            nxt_state,reward,done, score = env.play_move(action)
            episode_reward += reward
            episode_score += score
            nxt_q_values = Q_target.forward(convert_state_to_vec(nxt_state, env))[4]  # Get Q-values
            td_error = float(abs(reward + gamma * np.max(nxt_q_values) - Q.forward(q_state)[4][best_idx]))
            D.append(((q_state, full_state_cache), best_idx, reward, convert_state_to_vec(nxt_state, env), done, td_error))
            # Learining Phase

            if len(D) >= batch_size:
                minibatch, indices = sample(D, batch_size= batch_size)
                loss = learn(minibatch, batch_size,indices, gamma = gamma,  alpha = lr)
                episode_losses.append(loss)
            
            steps+=1
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
            
                
            
            
        
    return Q.get_weights(), scores, rewards, average_losses









from itertools import product

# Define parameter grid
param_grid = {
    'alpha': [2.5e-4, 1e-3, 1e-4, 1e-5],
    'gamma': [0.9, 0.95, 0.90, 0.50],
    "epsilon": [0.001, 0.01, 0.1, 2.5],
    'epsilon_decay': [0.995, 0.99, 0.98, 0,90],
    "alpha_decay" : [1, 0.999, 0.995, 0.990],
    'C': [ 200 , 500, 1000, 2000],
    'batch_size': [16, 32, 64, 128]
}

# Store best config
best_params = None
best_score = -np.inf

# Run grid search
configs = list(product(*param_grid.values()))

for alpha, gamma, epsilon, epsilon_decay, alpha_decay, C, batch_size in tqdm(configs, desc="Testing Configurations --- Search progress:", total=len(configs)):
    print(f"\nTesting config: alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}, C={C}, batch_size={batch_size} and alpha_decay {alpha_decay}, epsilon {epsilon}")
    
    # Train with current parameters; adjust 'train' as needed.
    score = train(num_episode=100 batch_size=batch_size, C=C, ep=20, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay)[1][-1]
   
    
    # Save best config
    if score > best_score:
        best_score = score
        best_params = {'alpha': alpha, 'gamma': gamma, 'epsilon_decay': epsilon_decay, 'C': C, 'batch_size': batch_size, 'alpha_decay': alpha_decay, 'epsilon': epsilon}
print("\nBest parameters:", best_params)
print("Best score:", best_score)



