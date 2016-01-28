import numpy as np
from Question3 import gen_coefficients

def data_generator2(N,v):
	coVar = np.identity(48)
	mean = np.zeros(48)
	data = np.array([np.random.multivariate_normal(mean, coVar) for k in range(N)])
	y = np.array([(np.dot(v,data[i])+ np.random.normal(0, 4)) for i in range(len(data))])
	y = np.reshape(y,(N,1))
	x = data.T
	return x, y, v

#print data_generator2(2, gen_coefficients())[1]

""" 4(d) It is possible to create the feature, simply just take a vector consists of 
48 univariate normal distributions with mean 0, variance 1
"""