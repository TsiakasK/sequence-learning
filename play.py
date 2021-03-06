#!/usr/bin/python

from naoqi import ALProxy
import numpy as np 
import readchar
import time
from random import randint
import os 
import muse_pyliblo_server as mps
import matplotlib.pyplot as plt
import gc
import random 
import webbrowser


# initialize random seed
random.seed(time.time())

# NAO parameters
ROBOT_IP = "129.107.119.45" # ethernet
#129.107.119.45
#129.107.119.45
#ROBOT_IP = "169.254.228.115"

tts = ALProxy("ALTextToSpeech", ROBOT_IP, 9559)
memory = ALProxy("ALMemory", ROBOT_IP, 9559)
aup = ALProxy("ALAudioPlayer", ROBOT_IP, 9559)


# user info and folders
user = raw_input('Enter userID: ')
user_folder = "data/user_" + str(user) + '/'

if os.path.exists(user_folder):
	session_id = len(os.listdir(user_folder)) + 1
else:
	session_id = 1

os.makedirs(user_folder + "session_" + str(session_id)) 
intro = open(user_folder + "session_" + str(session_id) + "/intro", 'w')
server = mps.initialize(intro)
server.start()

# Robot Introduction 
tts.say("Hi! My name is Stewie!")
tts.say("Let's play a game!")
time.sleep(0.5)
tts.say("I will say a sequence of letters and you have to repeat it!")
time.sleep(0.5)
tts.say("After the sequence is completed, you will listen to a beep sound!")
time.sleep(0.8)
aup.playSine(1000, 100, 0, 1)
time.sleep(1.5)
tts.say("After the sound, you have to respond, by pressing the buttons in the correct order") 
time.sleep(0.5)
tts.say("Before each sequence, I will tell you, the difficulty level of the sequence.") 
time.sleep(0.5)
tts.say("Level one has three letters! Level two, has five!! Level three, has 7 letters and level four, has nine!") 
time.sleep(0.5)
tts.say("Please remember!! You need to give your response. After the beep sound!")
time.sleep(0.5)
tts.say("Use only one hand, and make sure, that each button is pressed properly!")
time.sleep(0.5)
tts.say("Let's try with an example!")
time.sleep(0.5)

# example
seq = ["b", "b", "a", "c", "a"]
tts.say("Level 2")
time.sleep(0.5)
for item in seq:
	time.sleep(0.5)
	tts.say(item)
aup.playSine(1000, 100, 0, 1)
sig2 = 1 
res = []	
while(sig2):
	res.append(readchar.readkey().lower())
	if len(res) == len(seq):
		sig2 = 0

time.sleep(0.7)
tts.say("Great! Are you ready to start the session?")
#time.sleep(0.5)
# end of example

intro.close()
server.stop()
server.free()

positive_success = ["That was great! Keep up the good work!", "Wow, you do really great! Go on!", "That's awesome! You got it! Keep going!", "Fantastic! You do great! Keep going!"]
positive_failure = ["Oh, that was wrong! But that's fine! Don't give up!", "Oh, you missed it! No problem! Go on!", "Oops, that was not correct! That's OK! Keep going!", "Oops, too close! Stay focused and you will do it!"]
negative_success = ["Ok, that was easy enough! Let's see again!", "Well, ok! Maybe you were lucky! Let's check the next one!", "OK, you got it! Was it random?? Let's try again.", "OK, I guess you made it! Maybe, it was an easy one!"]
negative_failure = ["Hmmm! I don't think you are paying any attention! Try harder!", "Hey! Are you there? Stay focused when I speak!", "Oh! was that wrong? Well, you actually need to pay attention!", "If you want to succeed, you need to pay attention!"]
			
rf = 0
server = []
s = 0 
correct = 0 
total = 0 

D = [3,5,7,9]
L = ('a','b','c')
Actions = ["Easy", "Medium", "Hard", "Positive Feedback", "Negative Feedback"]

dirname = 'data/user_' + user + '/session_' +  str(session_id) + '/' 
turn = 1
game = 1

# 12 sequences of predefined actions for data collection -- show distribution of difficulty levels
action_seqs = [[1,2,4,3,2,1,0,2,3,5,3,4,1,1,0,5,2,3,2,1,4,0,4,3,2], 
	       [0,1,2,4,2,3,5,1,4,0,0,2,4,3,5,2,1,4,0,1,3,5,4,1,0], 
	       [2,3,4,1,1,0,5,2,1,0,4,1,5,2,3,4,3,2,2,1,5,2,0,4,1], 				      		
	       [2,2,5,3,1,4,1,0,5,3,4,2,1,0,4,1,1,3,5,3,1,0,2,2,4], 
	       [2,4,3,2,1,0,4,1,5,0,4,0,1,5,2,4,2,3,5,1,4,2,3,5,3], 
	       [2,3,5,3,1,0,4,1,0,4,1,2,2,5,1,2,4,3,2,5,2,1,0,4,2], 
	       [0,1,2,3,4,2,0,5,1,4,1,0,5,2,3,5,1,2,3,4,2,1,5,1,2], 
	       [1,0,1,5,2,3,2,5,1,0,4,1,5,2,1,0,5,2,3,4,2,3,2,4,1], 
	       [0,1,2,4,3,5,2,1,0,4,1,5,2,3,5,0,1,2,4,2,3,4,0,1,1], 
	       [1,0,2,3,4,1,0,1,5,2,4,3,5,2,1,0,2,4,3,5,2,1,4,2,3], 
	       [0,1,4,2,3,4,1,0,4,2,0,5,1,2,3,4,2,0,5,1,0,2,5,2,1], 
	       [0,1,2,5,1,0,4,2,3,5,3,1,5,0,4,2,5,2,3,2,1,4,3,2,3]]


#sessID = randint(0,11)
seqID = int(raw_input("Enter sequence ID: "))
actions = action_seqs[seqID]
print "seqID = " + str(seqID)

ps = 0 
total_score = 0 
previous_score = 0 
out = open(dirname + "/output", 'w')
log = open(dirname + "/state_EEG", 'a')
log2 = open(dirname + "/logfile", 'a')
server = mps.initialize(out)
server.start()

while (turn<=len(actions)): 	
	if not os.path.exists(dirname):
		os.makedirs(dirname) 
		
	response = []
	res = []
	rf = 0

	# select action from predefined
	action = actions[turn-1]

	## record EEG signals when robot announces the sequence ##
	out = open(dirname + "/robot_" + str(turn), 'w')
	server.f = out 
	##########################################################

	if action == 4: 
		rf = 1
		seq = list(np.random.choice(L, Dold))
	elif action == 5: 
		rf = 2
		seq = list(np.random.choice(L, Dold))
	else: 
		seq = list(np.random.choice(L, D[action]))
		Dold = D[action]

	length = len(seq)
	
	r = randint(0,3)
	if rf == 1: 
		if previous_success == 1: 
			tts.say(positive_success[r])
		else: 
			tts.say(positive_failure[r])

	if rf == 2: 
		if previous_success == 1: 
			tts.say(negative_success[r])
		else: 
			tts.say(negative_failure[r])

	# announce difficulty level
	time.sleep(0.5)
	diff_level = "Level" + str(D.index(length)+1)
	tts.say(diff_level)
	time.sleep(0.5)
	
	#announce sequence
	for item in seq:
		time.sleep(0.8)
		tts.say(item)

	aup.playSine(1000, 100, 0, 1)

	time.sleep(1)

	## record EEG signals when user presses the buttons ##
	out = open(dirname + "/user_" + str(turn), 'w')
	server.f = out
	######################################################

	# start time to measure response time
	start_time = time.time()

	################### CHECK USER RESPONSE AND CALCULATE SCORE ####################
	sig2 = 1
	first = 0 	
	while(sig2):
		res.append(readchar.readchar().lower())
		if first == 0: 
			reaction_time = time.time() - start_time
			first = 1
		if len(res) == Dold:
			sig2 = 0 
	
	completion_time = time.time() - start_time
	if seq != res:
		success = -1
		score = -1*(D.index(length)+1)
	else: 
		success = 1
		score = D.index(length) + 1
		correct += 1
	#################################################################################

	print "Turn No. " + str(turn) + " diff: " + str(diff_level) + " sequence: " +  str(seq) + " user's: " + str(res)

	dataline = str(length) + ' ' + str(rf) + ' ' + str(previous_score) + ' robot_' + str(turn) + ' user_' + str(turn) + '\n'	
	log.write(dataline)
	dataline = str(turn) + ' ' + str(length) + ' ' + str(rf) + ' ' + str(score) + ' ' + str(success) + ' ' + str(reaction_time) + ' ' + str(completion_time) + ' ' + str(seq) + ' ' + str(res) + '\n'
	log2.write(dataline)

	previous_success = success
	previous_score = score
	previous_rf = rf

	total_score += score
	turn += 1
		
out.close()
log.close()	
log2.close()
server.stop()

tts.say("That's the end of our session! Please take some time to complete a survey!")
time.sleep(0.5)
tts.say("Thank you for your time!! Hope to see you again!!!")

url = "https://form.jotform.us/72536243026148"
webbrowser.open_new(url)
