#!/usr/bin/python3

import numpy as np

# Set the environment size
ENV_SIZE = 5

class GridWorld():
    '''
    A simple 5x5 GridWorld environment for Value Iteration.
    The goal is to reach (4, 4) with the lowest cost (negative rewards).
    '''

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function
        self.V = np.zeros((env_size, env_size))
        
        # Define the terminal state
        self.terminal_state = (env_size - 1, env_size - 1) # (4, 4)
        self.V[self.terminal_state] = 0.0

        # Define the transition probabilities and rewards
        # Actions: (di, dj) for movement
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor (undiscounted problem)
        self.reward = -1.0  # Reward for non-terminal states
        self.theta = 1e-4 # Convergence threshold for Value Iteration
        
        # Policy variables
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
        # pi_str stores the optimal action descriptions (handles ties by showing multiple)
        self.pi_str = [["" for _ in range(env_size)] for _ in range(env_size)]
    
    '''@brief Returns True if the state is a terminal state
    '''
    def is_terminal_state(self, i, j):
        return (i, j) == self.terminal_state
    
    '''
    @brief Overwrites the current state-value function with a new one
    '''
    def update_value_function(self, V):
        self.V = np.copy(V)

    '''
    @brief Returns the full state-value function V_pi
    '''
    def get_value_function(self):
        return self.V

    '''@brief Returns the stored greedy policy (index of optimal action)
    '''
    def get_policy(self):
        return self.pi_greedy
    
    '''@brief Prints the policy using the action descriptions
    '''
    def print_policy(self):
        print("\nOptimal Policy (Direction):")
        for row in self.pi_str:
            print(row)

    '''@brief Calculate the maximum value for the current state (s)
              using the Bellman Optimality Equation:
              V*(s) = max_a (R(s, a) + gamma * V*(s'))
    '''
    def calculate_max_value(self, i, j):
        # The Bellman equation is applied only to non-terminal states
        if self.is_terminal_state(i, j):
            return 0.0, -1, "T" 

        max_value = float('-inf')
        best_action_index = -1
        best_actions_str = [] # To store actions in case of ties

        # Loop over all possible actions
        for action_index in range(len(self.actions)):
            # Find Next state
            next_i, next_j = self.step(action_index, i, j)

            # Check for boundary condition
            if self.is_valid_state(next_i, next_j):
                # If valid, use the value of the next state V[s']
                # Q(s, a) = R + gamma * V[s']
                current_value = self.reward + self.gamma * self.V[next_i, next_j]
            else:
                # If invalid (boundary), the agent stays in the current state (i, j)
                # Q(s, a) = R + gamma * V[s]
                current_value = self.reward + self.gamma * self.V[i, j]

            # Update the max_value
            if current_value > max_value:
                max_value = current_value
                best_action_index = action_index
                best_actions_str = [self.action_description[action_index]]
            elif current_value == max_value:
                # Handle ties by adding the action description
                best_actions_str.append(self.action_description[action_index])

        # Policy string for display: join tied actions with a slash
        policy_str = "/".join(best_actions_str)

        # Return the max value, the index of one of the best actions, and the policy string
        return max_value, best_action_index, policy_str
    
    '''@brief Returns the next state given the chosen action and current state
    '''
    def step(self, action_index, i, j):
        action = self.actions[action_index]
        return i + action[0], j + action[1]
    
    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''
    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid
    
    '''@brief Determines the optimal greedy policy based on the current optimal V function
    '''
    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if self.is_terminal_state(i, j):
                    self.pi_greedy[i, j] = -1 # Sentinel value
                    self.pi_str[i][j] = "T" # Terminal
                    continue

                # The V in self.calculate_max_value is the final converged V*
                _, best_action_index, best_actions_str = self.calculate_max_value(i, j)

                # Store the index of the best action
                self.pi_greedy[i, j] = best_action_index

                # Store the string representation of the best action(s)
                self.pi_str[i][j] = best_actions_str
        

# --- Main Value Iteration Execution ---

gridworld = GridWorld(ENV_SIZE)
num_iterations = 1000
theta = gridworld.theta # Convergence threshold

print(f"Starting Value Iteration with threshold theta = {theta}...")

for i in range(num_iterations):
    # Make a copy of the current value function (V_k)
    V_old = gridworld.get_value_function()
    V_new = np.copy(V_old) # Start the new V (V_{k+1}) as a copy of the old one
    
    max_delta = 0.0

    # Policy Evaluation/Improvement Step (Bellman Optimality Update)
    for row in range(ENV_SIZE):
        for col in range(ENV_SIZE):
            if gridworld.is_terminal_state(row, col):
                continue
            
            # Use the calculate_max_value function, which internally uses V_old (stored in self.V)
            max_value, _, _ = gridworld.calculate_max_value(row, col)
            
            # Update V_new
            V_new[row, col] = max_value

            # Calculate the maximum change across all states
            delta = abs(V_old[row, col] - V_new[row, col])
            max_delta = max(max_delta, delta)
            
    # After updating all states, update the value function in the environment object
    gridworld.update_value_function(V_new)
    
    # Check for convergence (stopping criteria)
    if max_delta < theta:
        print(f"Value Iteration converged after {i + 1} iterations.")
        break

if i == num_iterations - 1:
    print(f"Value Iteration finished after {num_iterations} iterations (max iterations reached).")

# Print the optimal value function
print("\nOptimal Value Function (V*):")
# Use numpy.round to display values clearly
print(np.round(gridworld.get_value_function(), decimals=2))

# Derive and print the optimal policy
gridworld.update_greedy_policy()
gridworld.print_policy()
