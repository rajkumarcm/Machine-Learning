"""
Author: Rajkumar Conjeevaram Mohan
Date: 27.04.2018
Program: K-Means Algorithm
"""

import numpy as np

def get_distance(v1,v2):
    temp = v1 - v2
    return np.sqrt(np.dot(temp,temp.T))

def kmeans(data,K):
    [N,M] = data.shape
    centroid_indices = np.zeros([K])
    membership = np.zeros([N])
    centroid = np.zeros([K,M])
    distance = np.zeros([N,K])
    width = np.zeros([K])
    epoch = 0
    # Initialise each cluster with a
    # random input object
    for k in range(K):
        index = np.random.randint(0, N)
        if any(centroid_indices != index):
            centroid_indices[k] = index
            centroid[k] = data[index]

    while True:
        changes = 0
        if epoch > 0:
            # Re-estimate the cluster centroids based
            # on the newly assigned data objects
            for k in range(K):
                centroid[k] = np.mean(data[np.where(membership==k)],axis=0)

        # Calculate the distance between data objects
        # and the initialised/re-estimated centroids
        for i in range(N):
            for k in range(K):
                # temp = data[i] - centroid[k]
                distance[i,k] = get_distance(data[i],centroid[k])

        # Assign each object to the closest cluster based
        # on the computed distance
        for i in range(N):
            k = np.argmin(distance[i])
            if membership[i] != k:
                membership[i] = k
                changes += 1

        # print("Epoch: %d, changes: %d"%(epoch,changes))

        if changes == 0:
            break

        epoch += 1

    group = []
    for k in range(K):
        # Grouping data objects into clusters
        # based on the estimated membership
        data_temp = data[np.where(membership == k)]
        group.append(data_temp)

        # Estimate the width of each cluster
        P = data_temp.shape[0]
        dist_temp = np.zeros([P,P])

        # Get the distance between data points
        # within each cluster, so that those
        # that are with highest difference
        # is considered the width or diameter
        # of the cluster
        for i in range(P):
            for j in range(P):
                dist_temp[i,j] = get_distance(data_temp[i],data_temp[j])

        # g and k are just the same
        width[k] = np.max(dist_temp)


    # print("\nPrinting centroid:")
    # for k in range(K):
    #     print(centroid[k])
    # print("\n")
    return centroid, width, group


# N = 100
# X = np.random.normal(loc=0,scale=0.1,size=[N,10])
# #print("Cluster 0:")
# #print(X)
# K = 10
# for k in range(1,K):
#      X2 = np.random.normal(loc=k,scale=k,size=[N,10])
#      X = np.vstack((X,X2))
# #X2 = np.random.normal(loc=5,scale=0.2*K,size=[N,5])
# #print("\nCluster 1:")
# #print(X2)
#
# #X = np.vstack((X,X2))
# np.random.shuffle(X)
# centroid,width,_ = kmeans(X,K)