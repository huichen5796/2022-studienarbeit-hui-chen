from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

points = [(1452, 370, 71, 15),
(1239, 370, 70, 15),
(834, 370, 363, 15),
(717, 370, 81, 15),
(572, 370, 78, 15),
(432, 370, 80, 15),
(289, 370, 78, 15),
(177, 370, 68, 15),
(39, 370, 112, 15),
(41, 310, 112, 15),
(1452, 298, 71, 15),
(1239, 298, 70, 15),
(834, 298, 363, 15),
(717, 298, 81, 15),
(572, 298, 78, 15),
(432, 298, 80, 15),
(289, 298, 78, 15),
(177, 298, 68, 15),
(77, 286, 36, 18),
(1453, 229, 70, 14),
(1239, 229, 70, 14),
(834, 229, 363, 14),
(717, 229, 81, 14),
(572, 229, 78, 14),
(432, 229, 80, 14),
(289, 229, 78, 14),
(178, 229, 67, 14),
(39, 229, 112, 14),
(178, 182, 66, 14),
(39, 182, 110, 14),
(1452, 181, 71, 15),
(1239, 181, 70, 15),
(834, 181, 363, 15),
(717, 181, 81, 15),
(572, 181, 78, 15),
(432, 181, 80, 15),
(289, 181, 78, 15),
(831, 122, 366, 14),
(700, 112, 116, 13),
(555, 112, 114, 13),
(415, 112, 115, 13),
(272, 112, 114, 13),
(1413, 102, 55, 18),
(1141, 101, 33, 14),
(1045, 101, 33, 14),
(950, 101, 33, 14),
(857, 101, 33, 14),
(1235, 87, 75, 15),
(45, 86, 101, 14),
(1359, 67, 163, 18),
(961, 40, 103, 18),
(630, 40, 106, 18),
(353, 40, 105, 18),
(176, 40, 73, 14)]

# x
points_x = np.array([[p[0],0] for p in points])

plt.figure(figsize=(6,1))
plt.scatter(points_x[:,0], points_x[:,1])
plt.ylim(-0.1, 0.1)
plt.yticks(np.arange(-0.1, 0.1, 0.1))
ax = plt.gca()
ax.xaxis.tick_top()
plt.savefig("27.svg", bbox_inches='tight',pad_inches = 0)
plt.show()

def GetSC(k):
    clf = KMeans(k, n_init = 'auto')
    predict_x = clf.fit_predict(points_x)
    # print(predict_x)

    # plt.figure(figsize=(6,6))
    # plt.scatter(points_x[:,0], points_x[:,1], c=predict_x)
    # plt.show()

    SC = silhouette_score(points_x, predict_x)
    coef.append([k,SC])
    return SC
    
coef = []
k = 2
SC_0 = -1

while(True):
    SC = GetSC(k)
    if k<40:
        SC_0 = SC
        k +=1
    else:
        print(coef)
        coef_array = np.array(coef)
        plt.figure(figsize=(6,6))
        plt.scatter(coef_array[:,0], coef_array[:,1])
        plt.plot(coef_array[:,0], coef_array[:,1])
        plt.savefig("59.svg", bbox_inches='tight',pad_inches = 0)
        plt.show()
        break

best = coef[-2]
best_k = best[0]
clf = clf = KMeans(best_k, n_init = 'auto')
label_x = clf.fit_predict(points_x)
plt.figure(figsize=(6,1))
plt.scatter(points_x[:,0], points_x[:,1], c=label_x)
plt.ylim(-0.1, 0.1)
plt.yticks(np.arange(-0.1, 0.1, 0.1))
ax = plt.gca()
ax.xaxis.tick_top()
plt.savefig("68.svg", bbox_inches='tight',pad_inches = 0)
plt.show()
print(label_x)


# y
points_y = np.array([[0,p[1]] for p in points])
plt.figure(figsize=(1,3))
plt.scatter(points_y[:,0], points_y[:,1])
plt.xlim(-0.1, 0.1)
plt.xticks(np.arange(-0.1, 0.1, 0.1))
ax = plt.gca()
ax.invert_yaxis()
ax.xaxis.tick_top()
plt.savefig("73.svg", bbox_inches='tight',pad_inches = 0)
plt.show()

def GetSC(k):
    clf = KMeans(k, n_init='auto')
    predict_y = clf.fit_predict(points_y)
    # print(predict_y)

    # plt.figure(figsize=(6,6))
    # plt.scatter(points_y[:,0], points_y[:,1], c=predict_y)
    # plt.show()

    SC = silhouette_score(points_y, predict_y)
    coef.append([k,SC])
    # print(coef)
    return SC
    
coef = []
k = 2
SC_0 = -1

while(True):
    SC = GetSC(k)
    if k<40:
        SC_0 = SC
        k +=1
    else:
        print(coef)
        coef_array = np.array(coef)
        plt.figure(figsize=(6,6))
        plt.scatter(coef_array[:,0], coef_array[:,1])
        plt.plot(coef_array[:,0], coef_array[:,1])
        plt.savefig("105.svg", bbox_inches='tight',pad_inches = 0)
        plt.show()
        break

best = coef[-2]
best_k = best[0]
clf = KMeans(best_k, n_init='auto')
label_y = clf.fit_predict(points_y)
plt.figure(figsize=(1,3))
plt.scatter(points_y[:,0], points_y[:,1], c=label_y)
plt.xlim(-0.1, 0.1)
plt.xticks(np.arange(-0.1, 0.1, 0.1))
ax = plt.gca()
ax.invert_yaxis()
ax.xaxis.tick_top()
plt.savefig("118.svg", bbox_inches='tight',pad_inches = 0)
plt.show()
print(label_y)