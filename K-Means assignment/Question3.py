import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import csv

dataset = np.loadtxt('data')
clusters = []
def Chisquaredist(v1, v2):
	dist = 0
	for i in range(len(v1)):
		dist += ((v1[i]- v2[i])**2)/(v1[i]+v2[i])
	return dist


def Kmedoids(centers, dataset, which_cost):
	dist= []
	clust=[]
	for i in range(len(dataset)):
		distlist =[]
		for j in range(len(centers)):
			if which_cost == 0:
				distlist.append(np.linalg.norm(centers[j]-dataset[i])**2)
			else:
				distlist.append(Chisquaredist(centers[j], dataset[i]))
		dist.append(distlist)
		clust.append(distlist.index(min(distlist)))
	clusterindex = [np.where(np.asarray(clust) == i)[0] for i in range(len(centers))]
	sumdist=sum(dist[i][clust[i]] for i in range(len(dist)))
	#print clust
	#print sumdist
	update = 10 #arbitrary number just greater than exp(-4)
	newcenter = [0 for i in range(len(centers))]
	value = False
	while value == False:
		#calculate new clusters
		clusterindex = [np.where(np.asarray(clust) == i)[0] for i in range(len(centers))]
		#new center base on cost functions
		#num_cen_list = [2, 3, 4, 8, 16]
		newcenter = [centers[m] for m in range(len(centers))]
		for i in range(len(centers)):
			#print i
			#print clust
			#print newcenter
			total_dist_list = []
			for k in range(len(dataset)):
				if clust[k] == i:
					if which_cost ==0:
						total_dist_list.append(np.linalg.norm(newcenter[i] - dataset[k])**2)
					else:
						total_dist_list.append(Chisquaredist(newcenter[i], dataset[k]))
				#min_total_dist = [np.linalg.norm(centers[i]-dataset[k]) if clust[k] == i for k in range(len(dataset))]
			min_total_dist = sum(total_dist_list)
			#print min_total_dist
			#print min_total_dist
			for j in range(len(clust)):
				if clust[j] == i:
					total_dist_list_2 =[]
					for k1 in range(len(dataset)):
						if clust[k1] == i:
							if which_cost == 0:
								total_dist_list_2.append(np.linalg.norm(dataset[j] - dataset[k1])**2)
							else:
								total_dist_list_2.append(Chisquaredist(dataset[j],dataset[k1]))
					total_dist_to_other_medoid_member = sum(total_dist_list_2)
					#rint total_dist_to_other_medoid_member
					if total_dist_to_other_medoid_member < min_total_dist:
						#print 'foo'
						#print total_dist_to_other_medoid_member
						newcenter[i] = dataset[j]
						min_total_dist =total_dist_to_other_medoid_member
		newcenter = np.asarray(newcenter)
		#print newcenter
		if np.array_equal(newcenter, centers) == False:
			centers = np.asarray([newcenter[m] for m in range(len(newcenter))])
		else:
			value = True
		dist = []
		clust = []
		#print newcenter
		#print centers
		#compute objective function of that particular instance
		for i in range(len(dataset)):
			distlist =[]
			for j in range(len(centers)):
				if which_cost ==0:
					distlist.append(np.linalg.norm(centers[j]-dataset[i])**2)
				else:
					distlist.append(Chisquaredist(centers[j], dataset[i]))
			dist.append(distlist)
			clust.append(distlist.index(min(distlist)))
			sumdist=sum(dist[i][clust[i]] for i in range(len(dist)))

	global clusters 
	clusters = clust
	return [centers, sumdist]


#======Part B, max of data:
maxX1 = max(dataset.T[0])
maxX2 = max(dataset.T[1])
factorX1 = 1/(0.01 + maxX1)
factorX2 = 1/(0.01 + maxX2)
newdataset = np.asarray([[dataset[i][0]* factorX1, dataset[i][1]* factorX2, 1-0.5 * dataset[i][0]* factorX1 - 0.5*dataset[i][1]* factorX2] for i in range(len(dataset))])

#choose random centers (choose 1, saved the random list, random need not be without replacement)
cen_num_list1 = [2,3,4,8,16]
'''
random_list = np.arange(2000)
a = np.random.permutation(2000)
cen_num_list = [2, 3, 4, 8, 16]

total_num = [[np.random.permutation(2000)[i] for i in range(cen_num_list[j])] for j in range(len(cen_num_list))]
'''
saved_center_index_list = [[345, 729], [582, 1180, 976], [1520, 185, 1080, 590], [982, 1454, 1549, 1675, 778, 1567, 1791, 102], [1658, 1943, 1011, 1312, 771, 1768, 460, 1925, 1155, 1305, 1736, 506, 134, 1329, 558, 403]]
cen2 = np.asarray([newdataset[saved_center_index_list[0][i]] for i in range(len(saved_center_index_list[0]))])
cen3 = np.asarray([newdataset[saved_center_index_list[1][i]] for i in range(len(saved_center_index_list[1]))])
cen4 = np.asarray([newdataset[saved_center_index_list[2][i]] for i in range(len(saved_center_index_list[2]))])
cen8 = np.asarray([newdataset[saved_center_index_list[3][i]] for i in range(len(saved_center_index_list[3]))])
cen16= np.asarray([newdataset[saved_center_index_list[4][i]] for i in range(len(saved_center_index_list[4]))])
#print cen2
#print cen2
#print newdataset
#print Kmedoids(cen2,newdataset,1)


#plotting result
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

def plotX(X,cluster,label, colCl):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(*zip(*X), marker = '.', color = colGen(label))
	ax.scatter(*zip(*cluster), s=80, facecolors='none', edgecolors='r')
	plt.xlabel("First vector Value")
	plt.ylabel("Second vector Value")
	plt.show()

new_dataset_trimmed = np.asarray([newdataset.T[0], newdataset.T[1]]).T

#plot the stylized data, for kmedoid 2, 3, 4,8, 16
#change cen2,cen3,cen4, cen8,cen16 for both of the 2 points
#Kmedoids(centers,dataset,which_cost); which_cost is 0 for squared euclidean, cost is 1 for chisquare
centersresult = Kmedoids(cen16,newdataset,1)[0]
centersresult_trimmed = np.asarray([centersresult.T[0], centersresult.T[1]]).T
#print clusters
#print dataset
plotX(new_dataset_trimmed, np.squeeze(centersresult_trimmed), clusters,'r')

#print np.linalg.norm(newdataset[0] - newdataset[1])

