"""
Stephen Vu Xuan Kim Cuong - 1000646
ESD Class of 2016
Fall 2015 Machine Learning
SUTD
"""
import numpy as np
import matplotlib.pyplot as plt

#----------------A----- creating the matrix-----
x = np.random.multivariate_normal([1,3], [[1,0],[0,1]], 100)
#----------tranpose of X will be the (2,100) matrix
x = x.transpose() 
print x
#-------------B plotting
plt.plot(x[0,:], x[1,:], '.')
plt.show()
plt.clf()

#------------D--- Mirror Y factor-----------
A1 = np.matrix('-1 0; 0 1')
xMirrored = A1*x


plt.plot(xMirrored[0,:], xMirrored[1,:], '.', color = "blue")
plt.show()

#------------F--- Scale factor-----------
A2 = np.matrix('0.5 0; 0 1')
xScaled = A2*x


plt.plot(xScaled[0,:], xScaled[1,:], '.', color = "blue")
plt.show()


#----------H----- Rotate Factor
sinp4 = np.sin(np.pi/4)
cosp4 = np.cos(np.pi/4)
A3 = np.matrix([[cosp4, sinp4], [-sinp4, cosp4]])
xRotated = A3*x


plt.plot(xRotated[0,:], xRotated[1,:], '.', color = "blue")
plt.show()

#----------J----- MirrorXFactor------
A4 = np.matrix('1 0; 0 -1')
xMirrorX = A4*x


#----------K----- Composite mapping factor------
A5 = A2*A1*A4
print A5

xMap = A5*x

plt.plot(xMap[0,:], xMap[1,:], '.', color = "blue")
plt.show()