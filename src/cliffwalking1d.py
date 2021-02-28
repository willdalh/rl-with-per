import numpy as np
import random

class CliffWalking1D(object):
	def __init__(self, nodes):
		self.world = np.zeros(nodes)
		self.pos = 0
		self.action_space = ActionSpace(2)
		for i in range(nodes):
			self.world[i] = int(random.randint(0, 1))

	# Resets position of agent back to start
	def reset(self):
		self.pos = 0
		return self.pos

	def step(self, action):
		next_state = None
		reward = None
		done = False
		info = None
		if self.world[self.pos] == action: # Correct action
			self.pos += 1
			next_state = self.pos
			reward = 0
			if (self.pos == len(self.world)): # At goal
				reward = 1

				done = True
		else: # Agent fell of the cliff
			self.pos += 1
			next_state = self.pos
			reward = 0
			done = True
		return next_state, reward, done, info

	def render(self):
		arrow_string = " " * (2*self.pos) + "↓"
		print(arrow_string)

		world_string = ""
		for i in range(len(self.world)):
			world_string += str(int(self.world[i])) + " "
		print(world_string + "G")

	def close(self):
		del self

class ActionSpace(object):
	def __init__(self, n):
		self.n = n
		self.space = range(0, n)
	
	def sample(self):
		return random.sample(self.space, k=1)[0]
