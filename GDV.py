# -*- coding: utf-8 -*-

import numpy as np

def tempfun(num):
    arr=[]
    for i in range(num):
        arr.append(i)
    new_labels = np.zeros(np.array(arr).shape, dtype=np.int)
    for i, s in enumerate(np.unique(arr)):
    # and set the new label with the bit set, which we want to have
        new_labels[arr == s] = 1 << i
    
    
    
def discrimination_value(points, labels, metric='euclidean', reduce=None, sample_points=True):
    """
    Calculates the discrimination value for points belonging to different clusters.

    Parameters
    ----------
    points : ndarray
        the N points with D dimensions in the format (N,D)
    labels : ndarray
        the N labels for the points. Format (N)
    metric : string, optional
        the metric to use, e.g. 'euclidean', 'mahalanobis', ... default: 'euclidean'
        See scipy.spatial.distance.pdist for more information on how what metrices are implemented.
    reduce : int, float, optional
        to speed up the calculation, the calculation can be carried out on a subset of the dataset. If reduce is an
        integer, it specified how many points of the dataset are used. If it is a float, it is interpreted as a
        fraction of points to use.

    Returns
    -------
    discrimination_value : float
        the calculated distrimination value
    """
    from scipy.spatial.distance import pdist, squareform, euclidean

    # if reduce is given, draw a subset of the dataset
    if reduce is not None and sample_points is True:
        # if reduce is a float, interpret it as a fraction
        if reduce < 1:
            N = len(labels)
            reduce = np.sqrt(reduce) * N

        # draw X points from the dataset
        #indices = np.linspace(0, len(labels)-1, int(reduce)).astype(int)
        indices = np.random.randint(0, len(labels) - 1, int(reduce)).astype(int)
        points = points[indices]
        labels = labels[indices]

    # norm the points
    points = points.copy().astype("float")
    points -= np.mean(points, axis=0)
    points /= 2 * (np.std(points, axis=0) + 1e-9)

    # we change the label values to different bits, to have an easy way to denote a label that is a pair of old labels
    # label 1: 0001, label 2: 0010, label 3: 0100
    # distance of a point in label 1 to a point in label 3 should receive the label : 0101

    # therefore we initialize an empty array of type int
    new_labels = np.zeros(labels.shape, dtype=int)

    # iterate over all unique labels in the labels list
    for i, s in enumerate(np.unique(labels)):
        # and set the new label with the bit set, which we want to have
        new_labels[labels == s] = 1 << i
    # we also create a unique list of the new labels
    unique_new_labels = 1 << np.arange(i+1)
    # and a list of all pairs
    unique_pair_labels = squareform(unique_new_labels[:, None] | unique_new_labels[None, :], checks=False)

    if reduce is not None and not sample_points:
        # if reduce is a float, interpret it as a fraction
        if reduce < 1:
            N = len(labels)
            reduce = reduce * N**2
        # draw two times
        indices1, indices2 = np.random.randint(0, len(labels) - 1, (2, int(reduce)))
        point_distances = np.linalg.norm(points[indices1] - points[indices2], axis=-1)
        pair_labels = new_labels[indices1] | new_labels[indices2]
    else:
        # then we create a list of the labels for each pair in the condensed distance matrix that we will calculate with
        # the points
        pair_labels = squareform(new_labels[:, None] | new_labels[None, :], checks=False)

        # calculate the distances between all points
        point_distances = pdist(points, metric=metric)

    # get the mean distance in each cluster and take the mean of these mean values
    mean_intra_sum = np.mean([np.mean(point_distances[pair_labels == l]) for l in unique_new_labels])
    # get the mean distance of points belonging to different clusters and take the mean of these mean values
    mean_inter_sum = np.mean([np.mean(point_distances[pair_labels == l]) for l in unique_pair_labels])

    # calculate the difference between these sums and normalize it with the squared number of dimensions
    return (mean_intra_sum - mean_inter_sum) / np.sqrt(points.shape[1])


def generate_labels(clusters):
    disc_mat = np.vstack(clusters)
    labels = np.hstack([np.ones(c.shape[0])*i for i, c in enumerate(clusters)])

    return disc_mat, labels


# =============================================================================
# points = np.load('gdvPoints.npy')
# labels = np.load('gdvLabels.npy')
# 
# discrimination_value(points,labels)
# =============================================================================

# =============================================================================
# if __name__ == "__main__":
#     import time
#     # number of points
#     N = 1000
#     # number of dimensions
#     D = 256
#     # number of categories
#     C = 10
# 
#     # generate clusters with random variables
#     clusters = []
#     for i in range(C):
#         clusters.append(np.random.normal(i, 0.2, (N, D)))
#     # generate the labels
#     data, labels = generate_labels(clusters)
# 
#     # calculate the discrimination value
#     t = time.time()
#     disc_value = discrimination_value(data, labels)
#     dt = time.time() - t
#     print("discrimination value:", disc_value, "time:", dt, "s")
# 
#     # calculate the discrimination value using only 10% of the data
#     t = time.time()
#     disc_value = discrimination_value(data, labels, reduce=0.1)
#     dt = time.time() - t
#     print("discrimination value:", disc_value, "time:", dt, "s")
# 
#     # calculate the discrimination value using only 1% of the data
#     t = time.time()
#     disc_value = discrimination_value(data, labels, reduce=0.01, sample_points=False)
#     dt = time.time() - t
#     print("discrimination value:", disc_value, "time:", dt, "s")
# 
#     # calculate the discrimination value using only 0.1% of the data
#     t = time.time()
#     disc_value = discrimination_value(data, labels, reduce=0.001, sample_points=False)
#     dt = time.time() - t
#     print("discrimination value:", disc_value, "time:", dt, "s")
# 
# =============================================================================
