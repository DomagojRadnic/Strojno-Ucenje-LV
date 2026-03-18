import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)
mpg = data[:,0]
hp = data[:,3]
wt = data[:,5]
cly = data [:,1]
mask = cly == 6
plt.scatter(mpg,hp, s=wt*30)
print (cly.max())
print (cly.min())
print (cly.mean())
print ()
plt.show()  