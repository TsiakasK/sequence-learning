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

    	
	
	@make_method('/muse/elements/horseshoe', 'ffff')
    	def isgood_callback(self, path, args):
		f = self.f
		ch1, ch2, ch3, ch4 = args
		#f.write("h " + str(ch1) + ' ' + str(ch2) + ' ' + str(ch3) + ' ' + str(ch4) + '\n')
		print ch1, ch2, ch3, ch4

def initialize(f):
	try:
		server = MuseServer(f)
		return server
	except KeyboardInterrupt:
		print ("End of program: Caught KeyboardInterrupt")
	    	sys.exit()
	except ServerError, err:
		print "lalalla"
	    	print str(err)
	    	sys.exit()
    

if __name__ == "__main__":
	f = open('test', 'w')
	s = initialize(f)
	time.sleep(3)
	s.start()
    	time.sleep(10)
	s.stop()
