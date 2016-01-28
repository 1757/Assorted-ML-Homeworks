#import itertools as it
import numpy as np

def gen_coefficients():
	indexlist = [i for i in range(48)]
	index = np.random.permutation(indexlist)
	coeff = np.zeros(48)
	for i in range(12):
		coeff[index[i]] = np.random.uniform(0.6,1)
	for k in range(12,48):
		coeff[index[k]] = np.random.uniform(0,0.2)
	return coeff
#print gen_coefficients()