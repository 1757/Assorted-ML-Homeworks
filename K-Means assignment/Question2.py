import numpy as np
from Question1 import gensomemixturedata as gen
import itertools
import matplotlib.pyplot as plt
import csv
#from sklearn.cluster import KMeans

prob = np.matrix([[0.15], [0.3],[0.55]])
mean = np.matrix([[3,6,5.1], [3,3.6,9]])
var = np.matrix([[1,1,1],[1,0.5,1.5]])


def Kppinit(cen_number, dataset):
	firstchoice= np.random.choice(len(dataset),1)
	pointset = [dataset[firstchoice]]
	cen_number -= 1
	while cen_number>0:
		#minimum euclidean distance
		mindistlist = [min([np.linalg.norm(dataset[i]-pointset[j]) for j in range(len(pointset))]) for i in range(len(dataset))]
		#probability proportionate to euclidean distance
		probabilitylist = [mindistlist[i]/sum(mindistlist) for i in range(len(mindistlist))]
		nextchoice = np.random.choice(len(dataset), 1, p=probabilitylist)
		pointset.append(dataset[nextchoice])
		cen_number-=1
	return pointset
clusters = []

def Kmeans(cen_number, dataset, tolerance = 1e-4):
	#run Kpp seed
	centers = Kppinit(cen_number, dataset)
	dist= []
	clust=[]
	for i in range(len(dataset)):
		distlist =[]
		for j in range(len(centers)):
			distlist.append(np.linalg.norm(centers[j]-dataset[i])**2)
		dist.append(distlist)
		clust.append(distlist.index(min(distlist)))
	clusterindex = [np.where(np.asarray(clust) == i)[0] for i in range(len(centers))]
	sumdist=sum(dist[i][clust[i]] for i in range(len(dist)))
	#print clust
	#print sumdist
	update = 10 #arbitrary number just greater than exp(-4)
	count = 0
	while update > tolerance:
		old = sumdist
		clusterindex = [np.where(np.asarray(clust) == i)[0] for i in range(len(centers))]
		#print clusterindex
		centers = [np.matrix([np.average([dataset[clusterindex[i][j],0] for j in range(len(clusterindex[i]))]), np.average([dataset[clusterindex[i][j],1] for j in range(len(clusterindex[i]))])]) for i in range(len(clusterindex))]
		dist= []
		clust=[]
		for i in range(len(dataset)):
			distlist =[]
			for j in range(len(centers)):
				distlist.append(np.linalg.norm(centers[j]-dataset[i])**2)
			dist.append(distlist)
			clust.append(distlist.index(min(distlist)))
		sumdist=sum(dist[i][clust[i]] for i in range(len(dist)))
		count +=1
		update = old-sumdist
		#print update
	#print distandclust
	#print update
	#print count
	global clusters 
	clusters = clust
	return [centers, sumdist]


#dataset = gen(2000, prob, mean, var)
#np.savetxt('data', dataset)
dataset = np.loadtxt('data')
#print dataset[0,1]

#plotting clusters

#Generate soft random color list (unecessary, just for the lolz)
colist = [(3*np.random.uniform()/4, 3*np.random.uniform()/4,3*np.random.uniform()/4) for i in range(16)]

def colGen(labVect):
	colArray=[]
	blue = 1.0
	green = 1.0
	red = 1.0
	for k in range(len(labVect)):
		colArray.append(colist[labVect[k]])
	return colArray

#uncomment this block of code to plot 
'''
def plotX(X,cluster,label, colCl):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(*zip(*X), marker = '.', color = colGen(label))
	ax.scatter(*zip(*cluster), marker = 'o', color = colCl)
	plt.xlabel("First vector Value")
	plt.ylabel("Second vector Value")
	plt.show()


#plot the stylized data, for kmean 2, 3, 4,8, 16
cen2 = Kmeans(8, dataset)[0]
print clusters
print dataset
plotX(dataset, np.squeeze(cen2), clusters,'r')

'''


#Compute and save Average for 2,3, 4, 8, 16 clusters (change the number to 2,3,4,8,16)
#the saved clusters and objective function will be appended in repData.txt
#in the sequence: 10 tries of 2, 3, 4, 8, 16 clusters respectively
#the data stored is corresponded to 'allAV1.txt'
'''
clusteravg2 = []
for k in range(10):
	m = Kmeans(16, dataset) #change the param for Kmeans here
	comb2 = list(itertools.combinations(m[0],2)) #pairwise combination of the data saved
	avgdist = sum([(np.linalg.norm(comb2[i][0]-comb2[i][1])**2) for i in range(len(comb2))])/len(comb2)
	#print avgdist
	clusteravg2.append(avgdist)
	#with open('repData.csv','a') as f_handle:
	#	f_handle.write(str(m)+"\n")
avg= sum(clusteravg2)/len(clusteravg2)

#append the result to the file allAV1 and allAV2, for AV2 there is no corresponding repeated data file (repdata.csv)
with open("allAV2.txt", "a") as myfile:
	myfile.write(',')
	myfile.write(str(avg))
'''
#plot the average value as a function of k
'''
with open('allAV1.txt', 'rb') as f:
    reader = csv.reader(f)
    AV1 = list(reader)
AV1 = AV1[0]

with open('allAV2.txt', 'rb') as f:
    reader = csv.reader(f)
    AV2 = list(reader)
AV2 = AV2[0]

AV1=[float(AV1[i]) for i in range(len(AV1))]
AV2=[float(AV2[i]) for i in range(len(AV2))]
k = [2,3,4,8,16]
#print AV1

plt.figure().suptitle('blue is first 10 tries average, red is second 10 tries')
plt.xlabel('number of clusters')
plt.ylabel('average square distance')
plt.plot(k, AV1)
plt.plot(k, AV2, color = 'r')
plt.show()

'''




# [['This is the first line', 'Line1'],
#  ['This is the second line', 'Line2'],












