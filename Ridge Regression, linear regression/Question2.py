import numpy as np

def data_generator1(N): 
	coVar = np.identity(48)*np.asarray([1.0/(i+1.0) for i in range(48)])
	mean = np.zeros(48)
	data = np.array([np.random.multivariate_normal(mean, coVar) for k in range(N)])
	y = np.array([(np.dot(np.ones(48),data[i])+ np.random.normal(0, 4)) for i in range(len(data))])
	y = np.reshape(y,(N,1))
	x = data.T
	return x , y

#k=data_generator1(2)[1]
#print np.dot(k,k.T).shape

""" (d) Yes it is possible to create this feature from 48 one-dimensional draw 
since the covariance matrix is diagonal, therefore we only need to generate 48
univariate normal distributions with mean 0, variance (1/(i+1)), i = 0-47"""
