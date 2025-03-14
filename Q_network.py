import numpy as np
import random
import pygame
import numpy as np
from collections import deque
import random
from collections import deque
import copy
from copy import deepcopy
from activations import *
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
        print(input.shape, self.W1.shape)
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
        