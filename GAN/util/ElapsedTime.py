#%%
import time

class ElapsedTime(object):
	"""
	A simple TimerObject to calculate the duration of various things.
	"""
	def __init__(self):
		self.begin = time.time()

	def getTime(self):
		"""
		Returns time since objectcreation in seconds.
		"""
		self.end = time.time()
		return self.end - self.begin

	def printTime(self):
		"""
		Prints duration since objectcreation.
		"""
		self.end = time.time()
		print(self.end - self.begin, 'seconds')

if __name__ == "__main__":
	Timer = ElapsedTime()
	time.sleep(3)
	Timer.printTime()