#!/usr/bin/python
import numpy as np 
from RL import MDP
from RL import Policy
from RL import Learning
from RL import Representation
import random
import matplotlib.pyplot as plt
import sys, getopt, os
import datetime
import time
import csv
import gc
from random import randint
import itertools
#from six.moves import cPickle
import pickle
import random
from datetime import datetime
from options import GetOptions

# define state-action space
def state_action_space():	
	length   = [3,5,7,9]
	feedback = [0,1,2]
	previous = [-4,-3,-2,-1,0,1,2,3,4]
	
	combs = (length, feedback, previous)
	states = list(itertools.product(*combs))
	states.append((0,0,0))
	
	l = [1, 2, 3, 4]
	f = [[1,0,0], [0,1,0], [0,0,1]]
	combs = (l, f, previous)
	normalized_states = list(itertools.product(*combs))

	actions = [0,1,2,3,4,5]
	actions_oh = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]
	return states, normalized_states, actions, actions_oh

def get_engagement(state, model, action_oh):
	#actions = {0:0, 1:0.2, 2:0.4, 3:0.6, 4:0.8 , 5:1.0 }
	#normalized_array =  normalized_states[states.index(tuple(state))]
	#normalized_array = list(normalized_array)
	#normalized_array.append(actions[action])
	X = np.asarray([state[0], state[1], state[2]])
	#X.reshape(-1, 1)
	engagement =  model.predict(X.reshape(1, -1))
	#print X
	#print concentration
	return engagement


def get_next_state(state, states, normalized_states, action, previous, model):
	levels = {3:1, 5:2, 7:3, 9:4}
	#normalized_state =  normalized_states[states.index(tuple(state))]
	#prob = model.predict(np.asarray(normalized_state).reshape(1,3), batch_size = 1)[0]
	#if random.random() <= prob: 
	#	success = 1
	#else: 
	#	success = -1

	#previous = success*levels[state[0]]
	if action == 0: 
		feedback = 0 
		length = 3
	if action == 1: 
		feedback = 0 
		length = 5
	if action == 2: 
		feedback = 0 
		length = 7
	if action == 3: 
		feedback = 0 
		length = 9
	if action == 4: 
		feedback = 1
		length = state[0]
	if action == 5: 
		feedback = 2
		length = state[0]
	
	next_state = [length, feedback, previous]
	normalized_next_state =  normalized_states[states.index(tuple(next_state))]
	st = normalized_next_state[0], normalized_next_state[1][0], normalized_next_state[1][1], normalized_next_state[1][2], normalized_next_state[2]
	prob = model.predict(np.asarray(st).reshape(1,5))[0]

	if random.random() <= prob: 
		success = 1
	else: 
		success = -1

	score = success*levels[length]
	if score < 0: 
		score = -1
	return score, [length, feedback, previous]


episodes, epochs, user, q, name, learn, interactive_type, To = GetOptions(sys.argv[1::])

if not os.path.exists('results/' + str(name)):
	os.makedirs('results/' + str(name))
g = open('results/' + name + '/episodes', 'w')
rr = open('results/' + name + '/return', 'w')
ss = open('results/' + name + '/score', 'w')
vv = open('results/' + name + '/mean_v(s)', 'w')
ee = open('results/' + name + '/engagement', 'w')

logfile = open('results/' + name + '/logfile', 'w')
logfile.write('logfile for: '  + str(name) + ' - ' + str(datetime.now()) + '\n\n')
logfile.write('Episodes: ' + str(episodes) + '\n')
logfile.write('Epochs: ' + str(epochs) + '\n')
logfile.write('User: ' + str(user) + '\n')
logfile.write('Qtable: ' + str(q) + '\n')
logfile.write('Learning: ' + str(learn) + '\n')
logfile.write('Interactive: ' + str(interactive_type) + '\n\n')
logfile.write('Exploration: ' + str(To) + '\n')
logfile.close()

#if int(user) > 0: 
#	f = open('user_models/user' + str(user) + '_engagement.model', 'rb')
#	cmodel = pickle.load(f)
#	f.close()

# score prediction network
f = open('user_models/user' + str(user) + '_performance.model', 'rb')
model = pickle.load(f)
f.close()
#raw_input()

# start and terminal states and indices
states, normed_states, actions, actions_oh = state_action_space()
A = ['L = 3', 'L = 5', 'L = 7', 'L = 9' , 'PF', 'NF']



first_length = random.choice([3,5,7,9])
start_state = (first_length,0,0)
start_state_index = states.index(tuple(start_state))

# define MDP and policy
m = MDP(start_state, actions)
m.states = states
#m.reward = [row[2] for row in m.states]
 
table = Representation('qtable', [m.actlist, m.states])
Q = np.asarray(table.Q)
if q: 
	print 'Loading Q-table: ' + str(q)
	ins = open(q,'r')
	Q = [[float(n) for n in line.split()] for line in ins]
	ins.close()	
table.Q = Q

egreedy = Policy('egreedy', To)
alpha = float(0.05)
gamma = float(0.9)
learning = Learning('qlearn', [alpha, gamma])

#episodes = 50000
#epochs = 100
episode = 1
R = []
V = []
S = []

ENG = []

while (episode < episodes): 
	start_state = (first_length,0,0)
	start_state_index = states.index(tuple(start_state))
	state_index = start_state_index
	state = start_state
	score = 0
	iteration = 1
	end_game = 0
	done = 0 	
	r = 0 
	quit_signal = 0 
	N = 20 
	previous_result = 0 
	EE = []
	random.seed(datetime.now())

	if episode % epochs == 0 or episode == 1:
		g.write('Episode No.' + str(episode) + '\n')
	if episode % 1000 == 0 or episode == 1:
		print episode, egreedy.param


	while(not done):
		state_index = states.index(tuple(state))
		egreedy.Q_state = Q[state_index][:]
		action = egreedy.return_action()
		result, next_state = get_next_state(state, states, normed_states, action, previous_result, model)
		
		next_state_index = states.index(tuple(next_state))
		reward = result
		
		#print state, action, next_state, reward
		
		engagement = 0
		#if next_state[1] > 0: 
		#	reward -= 4
		#reward = get_engagement(normed_states[next_state_index], cmodel)[0][0]
		#if int(user) > 0: 
		#	engagement = get_engagement(normed_states[next_state_index], cmodel)[0][0]

		###
		#reward = 0.6*result + 0.4*engagement
		###
		
		EE.append(engagement)

		#if interactive_type: 			
		#	reward += 0.5*(1.0 - engagement) 
	
		r += (learning.gamma**(iteration))*reward
		score += result

		if episode % epochs == 0 or episode == 1:
			g.write(str(iteration) + '... ' + str(state) + ' ' + str(A[action]) + ' ' + str(next_state) + ' ' + str(reward)  + ' ' + str(score)  + ' ' +  str(engagement) + '\n')

		if iteration == N:
			done = 1

		iteration += 1

		# reward shaping -- before update
		if interactive_type:
			reward += 0.9*engagement 
			
		## LEARNING 
		if learn: 
			Q[state_index][:] = learning.update(state_index, action, next_state_index, reward, Q[state_index][:], Q[next_state_index][:], done)
		# Q-augmentation -- after update
		#if interactive_type:
		#	Q[state_index][action] = 0.2*engagement

		state = next_state
		previous_result = result
		v_avg = np.asarray(Q).max(axis=1).mean()
		
	if egreedy.param > 0: 
		egreedy.param = To - (10*To*float(episode))/(6*float(episodes)) 
	
	if egreedy.param < 0: 
		egreedy.param = 0 
		print "no exploration at episode " + str(episode)  

	episode += 1
	R.append(r)
	V.append(v_avg)
	vv.write(str(v_avg) + '\n')
	rr.write(str(r) + '\n')
	ss.write(str(score) + '\n')
	ee.write(str(np.asarray(EE).mean()) + '\n')
	S.append(score)
	ENG.append(np.asarray(EE).mean())
	
with open('results/' + name + '/q_table', 'w') as f:
	writer = csv.writer(f,delimiter=' ')
	writer.writerows(Q)

tmp = []
return_epoch = []
for i, t in enumerate(R):
	tmp.append(t)
	if i%epochs == 0:
		a = np.asarray(tmp)
		return_epoch.append(a.mean())
		tmp = []

plt.plot(return_epoch)
plt.savefig('results/' + name + '/return.png')
plt.close()

tmp = []
eng_epoch = []
for i, t in enumerate(ENG):
	tmp.append(t)
	if i%epochs == 0:
		a = np.asarray(tmp)
		eng_epoch.append(a.mean())
		tmp = []

plt.plot(eng_epoch)
plt.savefig('results/' + name + '/engagement.png')
plt.close()

tmp = []
v_epoch = []
for i, t in enumerate(V):
	tmp.append(t)
	if i%epochs == 0:
		a = np.asarray(tmp)
		v_epoch.append(a.mean())
		tmp = []

plt.plot(v_epoch)
plt.savefig('results/' + name + '/mean_v(s).png')
plt.close()

tmp = []
score_epoch = []
for i, t in enumerate(S):
	tmp.append(t)
	if i%epochs == 0:
		a = np.asarray(tmp)
		score_epoch.append(a.mean())
		tmp = []

plt.plot(score_epoch)
plt.savefig('results/' + name + '/score.png')
plt.close()
