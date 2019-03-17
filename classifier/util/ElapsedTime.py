#%%
import time

class ElapsedTime(object):
	"""
	A simple TimerObject to calculate the duration of various things.
	"""
	def __init__(self):
		self.begin = time.time()

	def giveTime(self):
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
		print('Dauer: ',self.end - self.begin)

if __name__ == "__main__":
	Timer = ElapsedTime()
	time.sleep(3)
	Timer.printTime()