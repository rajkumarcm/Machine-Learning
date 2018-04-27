"""
Author: Rajkumar Conjeevaram Mohan
Date: 27.04.2018
Program: K-Means Algorithm
"""

import numpy as np

def kmeans(data,K):
    [N,M] = data.shape
    centroid_indices = np.zeros([K])
    membership = np.zeros([N])
    centroid = np.zeros([K,M])
    distance = np.zeros([N,K])
    epoch = 0
    # Initialise each cluster with a
    # random input object
    for k in range(K):
        index = np.random.randint(0, M)
        if not any(centroid_indices == index):
            centroid_indices[k] = index
            centroid[k] = data[index]

    while True:
        changes = 0
        if epoch > 0:
            # Restimate the cluster centroids based
            # on the newly assigned data objects
            for k in range(K):
                centroid[k] = np.mean(data[np.where(membership==k)],axis=0)

        # Calculate the distance between data objects
        # and the initialised centroids
        for i in range(N):
            for k in range(K):
                temp = data[i] - centroid[k]
                distance[i,k] = np.sqrt(np.dot(temp,temp.T))

        # Assign each object to the closest cluster based
        # on the computed distance
        for i in range(N):
            k = np.argmin(distance[i])
            if membership[i] != k:
                membership[i] = k
                changes += 1

        if changes == 0:
            break
        epoch += 1

    print("Printing cluster groups:")
    group = []
    for k in range(K):
        # print("Y cluster %d" % k)
        group.append(data[np.where(membership == k)])

    return centroid, group
    # print("\nPrinting centroid:")
    # for k in range(K):
    #     print(centroid[k])
    # print("\n")
    #


# X1 = np.random.normal(loc=0,scale=0.1,size=[10,5])
# X2 = np.random.normal(loc=10,scale=1,size=[10,5])
# X = np.vstack((X1,X2))
# np.random.shuffle(X)
# K = 2
# print("Cluster 0:")
# print(X1)
# print("\nCluster 1:")
# print(X2)
# kmeans(X,K)
