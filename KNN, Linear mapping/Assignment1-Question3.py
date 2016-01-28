"""
Stephen Vu Xuan Kim Cuong - 1000646
ESD Class of 2016
Fall 2015 Machine Learning
SUTD
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
#-------------------PART B: GENERATING AND PLOTTING DATA SETS--------------------------

#Generate binary label base on set probability
def binLab(vector):
	#throw a uniform [0,1) dice
	dice = np.random.random()

	#Creating label
	if vector[1] <= 0:
		if dice< 0.75:
			resLabel = -1
		else:
			resLabel = +1
	elif vector[1]> 0:
		if dice< 0.75:
			resLabel = +1
		else: 
			resLabel = -1
	return resLabel


#Generate color vector for plotting
def colGen(labVect):
	colArray=[]
	for k in range(len(labVect)):
		if labVect[k] == -1:
			colArray.append('b')
		else:
			colArray.append('r')
	return colArray



#plot the data set
def plotX(X, XLabel):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(*zip(*X), color = colGen(XLabel))
	ax.text(-0.75,1.25,'+1 Label is Red, -1 Label is Blue')

	ax.text(-.20, 0.25, '(+)', color = '#551a8b', fontsize= 30,
		bbox={'color':'yellow', 'alpha':0.7, 'pad':10})
	ax.text(-.20, -0.5, '( - )', color = '#551a8b', fontsize= 30,
	        bbox={'color':'yellow', 'alpha':0.7, 'pad':10})

	plt.xlabel("First vector Value")
	plt.ylabel("Second vector Value")
	plt.show()








#Generating training data base on uniform distribution
X2 = np.random.uniform(-1, 1 , size = (5000, 2))

#Creating the set of label for training data
X2Label = []
for k in range(len(X2)):
	X2Label.append(binLab(X2[k]))

print X2.shape

#Generating testing data
X3 = np.random.uniform(-1,1, size = (1000,2))

#Creating the set of label for training data
X3Label = []
for k in range(len(X3)):	
	X3Label.append(binLab(X3[k]))

##---- transposing X2 and X3, to match the requirement of the question
X2T = np.transpose(X2)
X3T = np.transpose(X3)


# This block is to test the distribution, to see whether the data is statistically correct
# count = 0 
# for k in range(len(X3Label)):
# 	if X3Label[k] == -1:
# 		count += 1
#print count

#plotting the data set 
plotX(X2, X2Label)
plotX(X3,X3Label)


#---------------------------C- KNN classifier------------------------

#function for KNN

#Euclidean distance of 2 points
def dist(x,y):
	return np.sqrt(np.sum(np.asarray((x-y))**2))

#K nearest neighbor algorithm (linear, bruteforce)
def knn(xtrain, ytrain, xtest, k):
	xtrain = np.transpose(xtrain)
	xtest = np.transpose(xtest)
	#xtrain, xtest has shape (2,number of data)
	ytest = []
	#loop through all test data
	for n in range(len(xtest)):	
		distList= []
		#for each test data, loop through all training data to find distances into distList
		#which corresponds to xtrain's indices
		for m in range(len(xtrain)):
			distList.append(dist(xtest[n] , xtrain[m]))
		#record the index of the k nearest neighbor, by popping the minimum distance's index
		sIndList = []
		for k1 in range(k):
			indeX = distList.index(min(distList))
			distList[indeX] = sys.maxint
			sIndList.append(indeX)
		#find the predicted label by adding all ytrain value of the index of k nearest neighbor
		#after that, classify the label by sign
		#for it to be impossible for sign = 0, k must not be the multiple of classes
		#in this case, k cannot be divisible by 2
		predictedLabel =0
		for k2 in sIndList:
			predictedLabel += ytrain[k2]
		if predictedLabel <=0:
			ytest.append(-1)
		else:
			ytest.append(+1)
	#ytest are the predicted label for xtest
	return ytest



#-----D--- Design 

#-----E--- Implementation, 0-1 error--------
X3LabTest=knn(X2T, X2Label,X3T,100)

print X3LabTest
print X3Label

error = 0
for k in range(len(X3Label)):
	if X3Label[k] != X3LabTest[k]:
		error +=1
print "0-1 error of data is %s%%" %(float(error)/(len(X3LabTest)) * 100)
plotX(X3, X3LabTest)

#------F, G ----- Scaling of first dimension, 0-1 error---------------
sFactor = np.matrix('1000 0; 0 1')
X2scaled  = X2*sFactor
print X2
print X2scaled
X3scaled = X3*sFactor


#transpose so that can use the KNN function
X2scaledT = np.transpose(X2scaled)
X3scaledT = np.transpose(X3scaled)

X3scaledLabTest = knn(X2scaledT, X2Label, X3T, 100)
error = 0
for k in range(len(X3Label)):
	if X3Label[k] != X3scaledLabTest[k]:
		error +=1
print "0-1 error of scaled data is %s%%" %(float(error)/(len(X3scaledLabTest)) * 100)
plotX(X3, X3scaledLabTest)

# --------H ---- Standard Deviation of training Data------
X2D1 = X2 * np.matrix('1;0')
X2D2 = X2 * np.matrix('0;1')
X2sd1 = np.std(X2D1)
X2sd2 = np.std(X2D2)


print "Standard Deviation of dimension 1 of X2 is %s" %X2sd1
print "Standard Deviation of dimension 2 of X2 is %s" %X2sd2

#----------I ---------- Normalizing data ------------

X3D1 = X3 * np.matrix('1;0')
X3D2 = X3 * np.matrix('0;1')
X3sd1 = np.std(X3D1)
X3sd2 = np.std(X3D2)



print "Standard Deviation of dimension 1 of X3 is %s" %X3sd1
print "Standard Deviation of dimension 2 of X3 is %s" %X3sd2

normfacX2 = np.matrix([[1/X3sd1, 0], [0, 1/X3sd2]])
normfacX3 = np.matrix([[1/X3sd1, 0], [0, 1/X3sd2]])

X2norm = X2*normfacX2
X3norm = X3*normfacX3
#transpose so that can use the KNN function
X2normT = np.transpose(X2norm)
X3normT = np.transpose(X3norm)
print np.std((X3norm * np.matrix('1;0')))


# print X2norm
# print X3norm

X3normLabTest = knn(X2normT, X2Label, X3normT, 100)
error = 0
for k in range(len(X3Label)):
	if X3Label[k] != X3normLabTest[k]:
		error +=1
print "0-1 error of scaled data is %s%%" %(float(error)/(len(X3normLabTest)) * 100)

plotX(X3, X3normLabTest)
#----------------