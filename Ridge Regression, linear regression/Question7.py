import numpy as np
from Question4 import data_generator2
from Question5 import train_ridgeregression
from Question3 import gen_coefficients

#---- Generate 500 samples:

lamd= np.exp(-2)
v = gen_coefficients()
total_sample= data_generator2(500,v )

#------------Holdout Protocol
def calculate_mean_square_error_HO(total_sample):
	indexlist = [i for i in range(500)]
	index = np.random.permutation(indexlist)
	trainingx = [total_sample[0].T[index[i]] for i in range(100)] 
	trainingy = [total_sample[1][index[i]] for i in range(100)]
	testingx = [total_sample[0].T[index[i]] for i in range(100,500)] 
	testingy = [total_sample[1][index[i]] for i in range(100,500)]
	training_result1 = train_ridgeregression(np.asarray(trainingx).T, trainingy, lamd)
	square_error = [(np.float(testingy[i]-np.dot(np.squeeze(training_result1),np.asarray(testingx)[i])))**2 for i in range(400)]
	return sum(square_error)/len(square_error)

errorlist = [calculate_mean_square_error_HO(total_sample) for i in range(10)]

A1 = sum(errorlist)/len(errorlist)
V1 = np.var(errorlist)

print A1
print V1

##--------------- 5-fold cross validation-----------
def generate_5fold(total_sample):
	indexlist = [i for i in range(500)]
	index = np.random.permutation(indexlist)
	x1 = [total_sample[0].T[index[i]] for i in range(100)] 
	y1 = [total_sample[1][index[i]] for i in range(100)]
	x2 = [total_sample[0].T[index[i]] for i in range(100,200)] 
	y2 = [total_sample[1][index[i]] for i in range(100,200)]
	x3 = [total_sample[0].T[index[i]] for i in range(200,300)] 
	y3 = [total_sample[1][index[i]] for i in range(200,300)]
	x4 = [total_sample[0].T[index[i]] for i in range(300,400)] 
	y4 = [total_sample[1][index[i]] for i in range(300,400)]
	x5 = [total_sample[0].T[index[i]] for i in range(400,500)] 
	y5 = [total_sample[1][index[i]] for i in range(400,500)]
	sets = []
	sets.append([[x1,y1], [x2+x3+x4+x5, y2+y3+y4+y5]])
	sets.append([[x2,y2], [x1+x3+x4+x5, y1+y3+y4+y5]])
	sets.append([[x3,y3], [x2+x1+x4+x5, y2+y1+y4+y5]])
	sets.append([[x4,y4], [x2+x3+x1+x5, y2+y3+y1+y5]])
	sets.append([[x5,y5], [x2+x3+x4+x1, y2+y3+y4+y1]])
	return sets

def calculate_mean_square_error_1fold(Tset):
	trainingx = Tset[0][0]
	trainingy = Tset[0][1]

	testingx = Tset[1][0]
	testingy = Tset[1][1]

	training_result1 = train_ridgeregression(np.asarray(trainingx).T, trainingy, lamd)
	square_error = [(np.float(testingy[i]-np.dot(np.squeeze(training_result1),np.asarray(testingx)[i])))**2 for i in range(400)]
	return sum(square_error)/len(square_error)

def calculate_mean_square_error_5fold(Tset):
	#print len(Tset)
	MSE_5f = [calculate_mean_square_error_1fold(Tset[i]) for i in range(len(Tset))]
	return sum(MSE_5f)/len(MSE_5f)

total_SE = [calculate_mean_square_error_5fold(generate_5fold(total_sample)) for i in range(10)]

A2 = sum(total_SE)/len(total_SE)
V2 = np.var(total_SE)
print A2
print V2
