import numpy as np
from Question2 import data_generator1
from numpy.linalg import inv
def train_ridgeregression(x,y,lmda):
	w = np.dot(np.dot(inv(np.dot(x, x.T) + lmda*(np.identity(np.dot(x,x.T).shape[0]))),x),y)
	return w
#print train_ridgeregression(data_generator1(10)[0],data_generator1(10)[1],0.1)