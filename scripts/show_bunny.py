import matplotlib.pyplot as plt
import numpy as np
import pickle

contourList = pickle.load(open('bunny_reconstructed.pickle', 'rb'))

x = []
y = []
z = []


def tolist(mat):
    l = []
    for val in mat:
        l.append(float(val))
    return l


for c in contourList:

    if np.shape(c)[1] > 0:
        x.extend(tolist(c[:,0]))
        y.extend(tolist(c[:,1]))
        z.extend(tolist(c[:,2]))


# Plot the two sets of coordinates
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z, 'r.')

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')

ax.set_xlim3d(-640, 640)
ax.set_ylim3d(-640, 640)
ax.set_zlim3d(-640, 640)


file = open('bunny.asc', 'w')
#file.write('x,y,z\n')
for i in range(0, len(x)):
    file.write(f'{x[i]} {y[i]} {z[i]}\n')
file.close()


plt.show()