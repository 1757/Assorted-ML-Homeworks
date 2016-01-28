import numpy as np


def gensomemixturedata(numdata, probs, means, sigmadiags):
	dist = np.random.uniform(0,1)
	covar = np.identity(2)
	x = [[-1, -1] for i in range(numdata)]
	if dist<prob[0]:
		covar= covar*np.asarray(sigmadiags[:,0])
		mean = np.squeeze(np.asarray(means[:,0]))
	elif dist < prob[1] + prob[0]:
		covar= covar*np.asarray(sigmadiags[:,1])
		mean = np.squeeze(np.asarray(means[:,1]))
	else:
		covar= covar*np.asarray(sigmadiags[:,2])
		mean = np.squeeze(np.asarray(means[:,2]))
	for i in range(numdata):
		while x[i][0]< 0 or x[i][1]<0:x[i] = np.random.multivariate_normal(mean, covar)
	return np.matrix(x)

#a = gensomemixturedata(10, prob, mean, var)

