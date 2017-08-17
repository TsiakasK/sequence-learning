from liblo import *
import datetime
import sys
import time
import gc
from random import randint

class MuseServer(ServerThread):
    	#listen for messages on port 5000
    	def __init__(self, f):
		self.f = f	
       	 	ServerThread.__init__(self, 5000)

	# receive horseshoe - status indicator values
	@make_method('/muse/elements/horseshoe', 'ffff')
    	def isgood_callback(self, path, args):
		f = self.f
		ch1, ch2, ch3, ch4 = args
		f.write("h " + str(ch1) + ' ' + str(ch2) + ' ' + str(ch3) + ' ' + str(ch4) + '\n')

    	#receive EEG data
    	@make_method('/muse/eeg', 'ffff')
    	def eeg_callback(self, path, args):
		f =  self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("eeg " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')
	
		
    	#receive elements data
    	@make_method('/muse/elements/alpha_relative', 'ffff')
    	def alpha_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("a " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

    	#receive elements data
    	@make_method('/muse/elements/beta_relative', 'ffff')
   	def beta_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("b " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

    	#receive elements data
    	@make_method('/muse/elements/gamma_relative', 'ffff')
    	def gamma_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("g " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')		

	#receive elements data
    	@make_method('/muse/elements/delta_relative', 'ffff')
   	def delta_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("d " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

    	#receive elements data
    	@make_method('/muse/elements/theta_relative', 'ffff')
   	def theta_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("t " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

	#receive elements data
    	@make_method('/muse/elements/alpha_absolute', 'ffff')
    	def alpha_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("Aa " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')
	
    	#receive elements data
    	@make_method('/muse/elements/beta_absolute', 'ffff')
   	def beta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("Ab " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

    	#receive elements data
    	@make_method('/muse/elements/gamma_absolute', 'ffff')
    	def gamma_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("Ag " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')		

	#receive elements data
    	@make_method('/muse/elements/delta_absolute', 'ffff')
   	def delta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("Ad " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

    	#receive elements data
    	@make_method('/muse/elements/theta_absolute', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("At " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

	#receive elements data
    	@make_method('/muse/elements/alpha_session_score', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("ds " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')

	#receive elements data
    	@make_method('/muse/elements/beta_session_score', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("ds " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')
		
	#receive elements data
    	@make_method('/muse/elements/gamma_session_score', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("ds " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')
	
	#receive elements data
    	@make_method('/muse/elements/delta_session_score', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("ds " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')
		
	#receive elements data
    	@make_method('/muse/elements/theta_session_score', 'ffff')
   	def theta_absolute_callback(self, path, args):
		f = self.f
		l_ear, l_forehead, r_forehead, r_ear = args
		f.write("ds " + str(l_ear) + ' ' + str(l_forehead) + ' ' + str(r_forehead) + ' ' + str(r_ear) + '\n')	
		
	#receive elements data
    	@make_method('/muse/elements/experimental/concentration', 'f')
    	def conc_callback(self, path, arg):
		f = self.f
		concentration = arg[0]
		f.write("c " + str(concentration) + '\n')		

def initialize(f):
	try:
		server = MuseServer(f)
		return server
	except KeyboardInterrupt:
		print ("End of program: Caught KeyboardInterrupt")
	    	sys.exit()
	except ServerError, err:
	    	print str(err)
	    	sys.exit()


if __name__ == "__main__":
	f = open('test', 'w')
	s = initialize(f)
	time.sleep(3)
	s.start()
    	time.sleep(5)
	s.stop()
