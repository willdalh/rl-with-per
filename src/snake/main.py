import gym
import gym_snake
from model import Net
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sys
import torchvision
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("../")
from utils import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

USE_TB = True
SAVE_MODEL = True


GAMMA = 0.98
ALPHA = 0.000075
MAX_MEMORY_SIZE = 70000
NUM_EPISODES = 20000
BATCH_SIZE = 64
REPLACE_AFTER = 100
USE_PER = sys.argv[1] == "True"

RUN_ID = ("GAMMA(%.2f)-ALPHA(%f)-MEMSIZE(%d)-EPISODES(%d)-BATCHSIZE(%d)-REPLACE_AFTER(%d)-PER(%r)" % (GAMMA, ALPHA, MAX_MEMORY_SIZE, NUM_EPISODES, BATCH_SIZE, REPLACE_AFTER, USE_PER))

grid_size = [8, 8]

time_started = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
if USE_TB: writer = SummaryWriter("runs-vs6/snake-%s-%s" % (RUN_ID, time_started))

def downsample(state):
    state = np.array(state, copy=True)
    state[state == np.array([0, 0, 255])] = 200 # Set food color
    downsampled = np.mean(state, axis=2, dtype=np.int16) # To grayscale
    downsampled[downsampled == 88] = 255 # Set snake-head-color
    return downsampled


env = gym.make("snake-v0")
env.unit_gap = 0
env.unit_size = 1
env.grid_size = grid_size

agent = Agent(Net=Net, input_size=1, output_size=env.action_space.n, gamma=GAMMA, alpha=ALPHA, max_memory_size=MAX_MEMORY_SIZE, num_episodes=NUM_EPISODES, replace_after=REPLACE_AFTER, use_per=USE_PER)

# Filling memory
print("Filling replay memory")
state = env.reset()
if USE_TB: 
    writer.add_graph(agent.Q_loc, torch.FloatTensor([downsample(state)]).to(device))
for t in range(2000):
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    agent.store_transition(downsample(state), action, reward, downsample(next_state), done)
    state = next_state
    if done:
        state = env.reset()
print("Finished filling memory")


# Learning
print("Started learning")

rewards = []
time_learned = []
n_episode = 100
time_per_n_episode = datetime.now()
for i_episode in range(NUM_EPISODES):
    stepcount = 0
    state = env.reset()
    state_d = downsample(state)
    done = False
    current_rewards = 0
    current_time_learned = []
    
    while not done:
        action = agent.choose_action(state_d, i_episode, train=True)
        next_state, reward, done, info = env.step(action)
        next_state_d = downsample(next_state)
        agent.store_transition(state_d, action, reward, next_state_d, done)
        state_d = next_state_d

        time_start = datetime.now()

        agent.learn(BATCH_SIZE, i_episode)

        time_difference = datetime.now() - time_start
        current_time_learned.append(time_difference.total_seconds())
        
        stepcount += 1
        current_rewards += reward
    
    if USE_TB: 
        writer.add_scalar("reward", current_rewards, i_episode)
        writer.add_scalar("steps done", stepcount, i_episode)
        writer.add_scalar("epsilon", agent.get_eps(i_episode), i_episode)
    rewards.append(current_rewards)
    time_learned.append(sum(current_time_learned)/len(current_time_learned))

    
    if (i_episode%n_episode == 0 and i_episode != 0):
        print("Using PER: " + str(USE_PER) + ". Learning at episode", str(i_episode)+"/"+str(NUM_EPISODES), "Average reward last 50 episodes: ", sum(rewards[-n_episode:])/n_episode, "Buffer size: ", agent.PER.tree.entries if USE_PER else len(agent.buffer.buffer), "Average time used learning:", sum(time_learned[-n_episode:])/n_episode)
        time_last_n_episodes = (datetime.now() - time_per_n_episode).total_seconds()
        print("Time used in the last %d episodes: "%(n_episode), time_last_n_episodes)
        if USE_TB: 
            writer.add_scalar("time used last %d episodes" % (n_episode), time_last_n_episodes, i_episode)
            writer.add_scalar("average reward last %d episodes" % (n_episode), sum(rewards[-n_episode:])/n_episode, i_episode)
        time_per_n_episode = datetime.now()


if SAVE_MODEL: torch.save(agent.Q_loc.state_dict(), "trainedmodels/vs6/trainedmodel-%s.pt" % (RUN_ID))


#Test
for i_episode in range(400):
    state = env.reset()
    done = False
    t = 0
    current_rewards = 0
    while not done:
        action = agent.choose_action(downsample(state), 100, train=False)
        new_state, reward, done, info = env.step(action)
        state = new_state
        t += 1
        current_rewards += reward
        if done or t == 100:
            break 
    if USE_TB: writer.add_scalar("rewards from testing", current_rewards, i_episode)
    
# Demo
for i_episode in range(0):
    state = env.reset()
    done = False
    t = 0
    print("Demo episode", i_episode)
    while not done:
        action = agent.choose_action(downsample(state), 100, train=False)
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        t += 1
        if done or t == 100:
            break 
env.close()

if USE_TB: writer.close()
