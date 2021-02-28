import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque


class SumTree(object):
    pointer = 0 # Position of next element in row of leaf nodes
    def __init__(self, cap):
        self.cap = cap
        self.tree = np.zeros(2 * cap - 1) # In total 2 * cap - 1 nodes
        self.data = np.zeros(cap, dtype=object) # Separate array for storing transitions
        self.entries = 0 # Count for stored transitions
    
    # Add new transition and priority to tree
    def add(self, priority, data):
        index = self.pointer + self.cap - 1 # Position of child node
        self.data[self.pointer] = data # Store the transition
        self.update(index, priority) # Update sums

        # Reset pointer back to start to replace elements when buffer is full
        self.pointer += 1 
        if self.pointer >= self.cap: 
            self.pointer = 0

        if self.entries < self.cap:
            self.entries += 1

    # Update parent node's values
    def update(self, index, priority):
        difference = priority - self.tree[index]
        self.tree[index] = priority # Set the new priority

        # Add the difference to the parent nodes
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += difference

    # Retrieve the node that has the input value in its interval
    # This process is described in the theory section
    def get_leaf(self, val):
        parent_index = 0
        while True:
            left_index = 2 * parent_index +1
            right_index = left_index + 1
            if left_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if val <= self.tree[left_index]:
                    parent_index = left_index
                else:
                    val -= self.tree[left_index]
                    parent_index = right_index
        data_index = leaf_index - self.cap + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    # Get the sum of all priorities
    def get_total_priority(self):
        return self.tree[0] # Value of root node


# Prioritized experience replay
class PrioritizedReplayBuffer:
    def __init__(self, cap, a=0.6, epsilon = 0.1):
        self.cap = cap
        self.a = a # Hyperparameter for degree of prioritization
        self.epsilon = epsilon # Bias
        self.tree = SumTree(cap) # Sumtree for storing transitions and priorities
    
    # Add transition to buffer
    def append(self, transition):
        max_priority = np.max(self.tree.tree[-self.cap:]) + abs(transition[2]) # + abs(reward)
        if max_priority == 0: # To avoid transitions with zero priority
            max_priority = 1
        self.tree.add(max_priority, transition) 
    
    # Update priority after training and calculating error of transition
    def update_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            priority = self.error_to_priority(e)
            self.tree.update(i, priority)

    # Convert error to priority
    def error_to_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.a

    # Calculate importance weights
    def get_importance(self, probabilites, b):
        importance = np.power(1/(self.tree.entries * probabilites), b)
        normalized = importance/max(importance) # Normalizing to not enlarge gradients
        return normalized

    # Fetches a batch of transitions from the buffer
    def get_samples(self, batch_size, b, device):
        if batch_size > self.tree.entries:
            batch_size = self.tree.entries

        indices = []
        priorities = []
        batch = []
        
        # Divide row of leaf nodes into segments
        segment_length = self.tree.get_total_priority() / batch_size

        # Fetch random transitions from each segment
        for i in range(batch_size):
            segment_start = segment_length * i
            segment_end = segment_length * (i + 1)
            
            # Preventing rare case of sampling empty node
            retries = 0
            while True:
                s = random.uniform(segment_start, segment_end) # Get random number in segment
                index, priority, transition = self.tree.get_leaf(s)
                if priority != 0:
                    break
                retries += 1
                # Discard if no non-empty leaf found after 5 attempts
                if retries == 5:
                    print("Stopped at 5 retries")
                    break
            if retries != 5: # Store non-empty element in batch
                indices.append(index)
                priorities.append(priority)
                batch.append(transition)
        
        probabilities = priorities / self.tree.get_total_priority() # Calculate probabilities
        importance =  self.get_importance(probabilities, b) # Get importance weights
        
        samples = np.array(batch, dtype=object)

        # Create tensors of transitions
        states = torch.tensor(samples[:, 0].tolist(), dtype=torch.float32).to(device)
        actions = torch.tensor(samples[:, 1].tolist(), dtype=torch.long).to(device)
        rewards = torch.tensor(samples[:, 2].tolist(), dtype=torch.float32).to(device)
        next_states = torch.tensor(samples[:, 3].tolist(), dtype=torch.float32).to(device)
        dones = torch.tensor(samples[:, 4].tolist(), dtype=torch.float32).to(device)
        return  states, actions, rewards, next_states, dones, np.array(indices), importance

    

# Non-prioritized experience replay
class ReplayBuffer:
    def __init__(self, cap):
        self.buffer = deque(maxlen=cap) # Using deque from collections library
        self.cap = cap

    # Append transition to buffer
    def append(self, transition):
        self.buffer.append(transition)

    # Fetches a batch of transitions from the buffer
    def get_samples(self, batch_size, device):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        # Sample random batch
        samples = np.array(random.choices(self.buffer, k=batch_size), dtype=object) 

        # Create tensors of transitions
        states = torch.tensor(samples[:, 0].tolist(), dtype=torch.float32).to(device)
        actions = torch.tensor(samples[:, 1].tolist(), dtype=torch.long).to(device)
        rewards = torch.tensor(samples[:, 2].tolist(), dtype=torch.float32).to(device)
        next_states = torch.tensor(samples[:, 3].tolist(), dtype=torch.float32).to(device)
        dones = torch.tensor(samples[:, 4].tolist(), dtype=torch.float32).to(device)
        return  states, actions, rewards, next_states, dones


class Agent(object):
    def __init__(self, Net, input_size, output_size, gamma, alpha, max_memory_size, num_episodes, replace_after, use_per, eps_min=0.05, per_a = 0.6):
        self.gamma = gamma
        self.eps_min = eps_min
        self.action_size = output_size
        self.action_space = range(output_size)
        self.learn_step_counter = 0
        self.replace_after = replace_after
        self.Q_loc = Net(input_size, output_size, alpha)
        self.Q_tar = Net(input_size, output_size, alpha)
        self.num_episodes = num_episodes
        self.use_per = use_per

        # Specify type of buffer
        if use_per:
            self.PER = PrioritizedReplayBuffer(max_memory_size, a=per_a)
        else:
            self.buffer = ReplayBuffer(max_memory_size)
        print("Using PER" if use_per else "Not using PER")

    # Return epsilon, used exploring and importance sampling (IS)
    def get_eps(self, episode):
        # Arbitrary function for calculating epsilon
        return max(self.eps_min, (1-4/(self.num_episodes))**episode)

    # Choose an action
    def choose_action(self, state, episode, train):
        action = None
        if train and np.random.random() < self.get_eps(episode): # train = exploring
            action = np.random.choice(self.action_space) # Choose random action for explring
        else: 
            state = torch.tensor([state], dtype=torch.float32).to(self.Q_loc.device)
            actions = self.Q_loc.forward(state) # Run the state through the local Q-Network
            action = torch.argmax(actions).item() # Get the index corresponding to the largest Q-value
        return int(action)

    # Replace the target network's parameters with the one's from the local network
    def replace_network(self):
        self.Q_tar.load_state_dict(self.Q_loc.state_dict())

    # Stores a transition in the used buffer
    def store_transition(self, state, action, reward, next_state, done):
        # Specify type of buffer to store transition in
        if self.use_per:
            self.PER.append([state, action, reward, next_state, done])
        else:
            self.buffer.append([state, action, reward, next_state, done])

    # Replay remembered transitions
    def learn(self, batch_size, episode, use_DDQN=True):
        self.Q_loc.optimizer.zero_grad()
        if self.learn_step_counter % self.replace_after == 0: # Replace target network every fixed number of learn-iterations
            self.replace_network()
        
        # Sample from the used buffer
        if self.use_per:
            # Beta goes from 0.4 to 1 throughout training
            states, actions, rewards, next_states, dones, indices, importance = self.PER.get_samples(batch_size, max(0.4, min(1, 1.05-self.get_eps(episode))), self.Q_loc.device)
        else:
            states, actions, rewards, next_states, dones = self.buffer.get_samples(batch_size, self.Q_loc.device)

        actions = actions.unsqueeze(1) # to shape(batch_size, 1)
        rewards = rewards.unsqueeze(1) # to shape(batch_size, 1)
        dones = dones.unsqueeze(1) # to shape(batch_size, 1)

        # Calculate predicted Q-values from states and gather the Q-value for each corresponding to the action taken
        Qpred = self.Q_loc.forward(states).gather(1, actions.long()).to(self.Q_loc.device) 
 
        with torch.no_grad():
            if use_DDQN:
                next_Qpred = self.Q_loc.forward(next_states).to(self.Q_loc.device) # Get Q-values for next states
                next_actions = torch.argmax(next_Qpred, dim=1) # Get the actions with the highest Q-values
                target = self.Q_tar.forward(next_states).to(self.Q_loc.device) # Calculate target Q-values from next states
                target = target.gather(1, next_actions.unsqueeze(1)) # Get Q-values 
                target = rewards + (1 - dones) * self.gamma * target # Bellman equation
            else:
                target = self.Q_tar.forward(next_states) # Get Q-values for next states
                target = rewards + (1 - dones) * self.gamma * torch.max(target, dim=1, keepdim=True)[0] # Bellman equation
        
        loss = None
        if self.use_per:
            error = Qpred - target
            errors_np = (torch.abs(error)).cpu().data.numpy()
            self.PER.update_priorities(indices, errors_np) # Update priorities for transitions in batch
            # Calculate loss with importance sampling (IS)
            importance_tensor = torch.FloatTensor(importance).to(self.Q_loc.device)
            loss = torch.mean((importance_tensor * torch.square(error)))
        else:
            loss = self.Q_loc.loss(Qpred, target) # Normal mean-squared-error
        
        loss.backward() # Calculate gradients
        self.Q_loc.optimizer.step() # Update network
        self.learn_step_counter += 1 
