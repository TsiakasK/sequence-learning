#!/usr/bin/python
import numpy as np 
import random as rnd

## TODO
# tabular, Qlearning, Sarsa, eligibility traces, actor critic, policy gradient, Q or V, QNN

#Return list of position of largest element  -- RANDOM between equals
def maxs(seq):	
	max_indices = []
	#if seq:
	max_val = seq[0]
	for i,val in ((i,val) for i,val in enumerate(seq) if val >= max_val):
		if val == max_val:
		       	max_indices.append(i)
		else:
				max_val = val
				max_indices = [i]

	return rnd.choice(max_indices)

def cdf(seq):
	r = rnd.random()
	print seq, r
	for i, s in enumerate(seq):
		if r <= s:
			return i

class MDP:
	def __init__(self, init, actlist, terminals= [], gamma=.9):
		self.init = init
		self.actlist = actlist
		self.terminals = terminals
		if not (0 <= gamma < 1):
		    raise ValueError("An MDP must have 0 <= gamma < 1")
		self.gamma = gamma
		self.states = set()
		self.reward = 0
		
	def actions(self, state):
		"""Set of actions that can be performed in this state.  By default, a
		fixed list of actions, except for terminal states. Override this
		method if you need to specialize by state."""
		if state in self.terminals:
		    return [None]
		else:
		    return self.actlist

class Policy:
	def __init__(self, name, param, Q_state = []):
		self.name = name
		self.param = param
		self.Q_state = Q_state
		
	def return_action(self):
		if self.name == 'softmax': 
			values = self.Q_state
			tau = self.param
			maxQ = 0 
			av = np.asarray(values)
	 		n = len(av)
			probs = np.zeros(n)
			
			for i in range(n):
				softm = ( np.exp(av[i] / tau) / np.sum( np.exp(av[:] / tau) ) )
				probs[i] = softm
				
			return np.where(np.random.multinomial(1,probs))[0][0]

		if self.name == 'egreedy': 
			values = self.Q_state
			maxQ = max(values)
			e = self.param
			if rnd.random() < e: # exploration
        			return rnd.randint(0,len(values)-1)
			else: 		     # exploitation
				return maxs(values)

class Representation:
	# qtable, neural network, policy function, function approximation
	def __init__(self, name, params):
		self.name = name
		self.params = params
		if self.name == 'qtable':
			[self.actlist, self.states] = self.params
			self.Q = [[0.0] * len(self.actlist) for x in range(len(self.states))] 
			 

class Learning:
	# qlearning, sarsa, traces, actor critic, policy gradient
	def __init__(self, name, params):
		self.name = name
		self.params = params
		if self.name == 'qlearn' or self.name == 'sarsa':
			self.alpha = self.params[0]
			self.gamma = self.params[1]
	
	def update(self, state, action, next_state, next_action, reward, Q_state, Q_next_state, done):
		if done: 
			Q_state[action] =  Q_state[action] + self.alpha*(reward - Q_state[action])
		else: 
			if self.name == 'qlearn':
				#print "qlearn"
				Q_state[action] +=  self.alpha*(reward + self.gamma*max(Q_next_state) - Q_state[action])
			if self.name == 'sarsa':
				#print "sarsa: Q[state][action]: "
				#print Q_state[action]
				learning =  self.alpha*(reward + self.gamma*Q_next_state[next_action] - Q_state[action])
				#print learning 				
				Q_state[action] = Q_state[action] + learning
				#print Q_state[action]
		return Q_state
		

"""
m = MDP([0,0], [0,1,2,3,4], 1)
m.states = [[0,0], [0,1], [1,1]]
print m.actlist, m.states
q = Representation('qtable', [m.actlist, m.states])
print q.Q
softmax = Policy('softmax', 1, [11.2, 2.2, 13.4, 12.3])
a = softmax.return_action()
print a
egreedy = Policy('egreedy', 0, [1.2, 2.2, 13.4, 12.3])
a = egreedy.return_action()
print a
"""

