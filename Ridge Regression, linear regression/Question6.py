import numpy as np
from Question5 import train_ridgeregression
from Question2 import data_generator1


#print len(training_sample[0].T)

lamd1= np.exp(-30)
lamd2= 5


def calculate_mean_square_error(trainsize, testsize, lamd):
	training_sample = data_generator1(trainsize)
	testing_sample = data_generator1(testsize)
	training_result1 = train_ridgeregression(training_sample[0], training_sample[1], lamd)
	square_error = [(np.float(testing_sample[1][i]-np.dot(np.squeeze(training_result1),testing_sample[0].T[i])))**2 for i in range(1000)]
	return sum(square_error)/len(square_error)

#-----------Error of lambda1, 100 training sample and lambda2, 100 training sample------
lam1error = [calculate_mean_square_error(100, 1000, lamd1) for i in range(10)]
meanlam1error = sum(lam1error)/10


lam2error = [calculate_mean_square_error(100, 1000, lamd2) for i in range(10)]
meanlam2error = sum(lam2error)/10



print meanlam1error
print meanlam2error
#-----------Error of lambda1, 500 training sample and lambda2, 500 training sample-----

lam1error2 = [calculate_mean_square_error(5000, 1000, lamd1) for i in range(10)]
meanlam1error2 = sum(lam1error2)/10


lam2error2 = [calculate_mean_square_error(5000, 1000, lamd2) for i in range(10)]
meanlam2error2 = sum(lam2error2)/10

print meanlam1error2
print meanlam2error2
#the result shows that at low training data sample, having a strong regularizer improves the fit
#however, if we have larger training data sample, we only need a small regularizer since the variance is reduced naturally due to the large data set