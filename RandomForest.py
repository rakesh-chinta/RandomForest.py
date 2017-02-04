import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassfier

xBlue = np.array([0,3,0,5,1,3])
yBlue = np.array([2,4,4,7,2,4])


xRed = np.array([4,6,2,8,2,5])
yRed = np.array([2,4,4,7,7,4])

X = np.array([0,2],[3,4],[0,4],[5,7],[1,2],[3,4],[4,2],[6,4],[2,4],[8,7],[2,7],[5,4])
y = np.array([0,0,0,1,1,1]) # 0:Blue class 1: Red class

plt.plot(xBlue,yBlue, 'ro', color = 'blue')
plt.plot(xRed, yRed, 'ro', color = 'red')

plt.plot(3,5 ,'ro', color = 'green', markersize=15)

plt.axis([-0.5,10,-0.5,10])

classifier = RandomForestClassfier(n_estimator = 100)
classifier.fit(X,y)

pred = classifier.predict([3,5])
print(pred)

plt.show()