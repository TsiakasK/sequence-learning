#!/usr/bin/python
import sys, getopt, os
from datetime import datetime

def GetOptions(argv): 
	learning = 1 
	interactive_type = 0
	name = datetime.now()
	episodes = 1000
	epochs = 10 
	user = 1 
	Table = 0
	exploration = 1
	OKGREEN = '\033[92m'
	ENDC = '\033[0m'

	try:
		opts, args = getopt.getopt(argv,"he:q:p:l:u:n:i:t:")
		#print opts, args
	except getopt.GetoptError:
		print '\n' + OKGREEN + 'USAGE:\n'
		print './sequence_learning.py -e episodes -p epochs -q qtable -u user -n name -l learning -i interactive_type -t exploration parameter ' + ENDC + '\n'
		sys.exit(2)
	for opt, arg in opts:
		#print opt, arg
		if opt == '-h':
			print '\n' + OKGREEN + 'USAGE:\n'
			print './sequence_learning.py -e episodes -p epochs -q qtable -u user -n name -l learning -i interactive_type\n'
			print "episodes in sumber of learning episodes (integer) -- default 5000"
			print "epochs is the number of episodes per epoch -- default 50"
			print "qtable is the name of the q_table file to load -- default is based on 'empty'"
			print "name is the name of the folder -- default is based on date"
			print "user is the user cluster (user1, user2) used for the experiment -- default 1"
			print "interactive_type is the selection of none (0), feedback (1), guidance (2), or both (3) -- default 0"
			print "learning: 0 for no learning and 1 for learning (Q-values update)-- default 1 \n\n" + ENDC
			sys.exit()
		elif opt in ("-q", "--qtable"):
		 	Table = arg
		elif opt in ("-t", "--exploration"):
		 	exploration = float(arg)
		elif opt in ("-e", "--episodes"):
		 	episodes = int(arg)
		elif opt in ("-u", "--user"):
		 	user = int(arg)
		elif opt in ("-p", "--epochs"):
		 	epochs = float(arg)
		elif opt in ("-n", "--name"):
		 	name = str(arg)
		elif opt in ("-i", "--interactive"):
		 	interactive_type = int(arg)
		elif opt in ("-l", "--learning"):
		 	learning = int(arg)

	if len(argv[1::]) == 0 :
		print '\n' + OKGREEN + 'Running with default parameters...' + ENDC + '\n'

	return episodes, epochs, user, Table, name, learning, interactive_type, exploration


#e, p, u, t, n, l, i = GetOptions(sys.argv[1::])
#print e, p, u, t, n, l, i
