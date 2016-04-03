'''
Anomaly detection exercise
by Stijn Geselle
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import cycle

def cluster_data(X, mu):
    clusters  = {}
    
    for x in X:
        norms = []
        
        # Iterate over all centroids and find the one with the minimum norm
        for centroid in enumerate(mu):
            norm = np.linalg.norm(x-mu[centroid[0]])
            norms.append( (centroid[0], norm) )
            
        bestcluster = min(norms, key=lambda t: t[1])[0]
        
        if bestcluster in clusters:
            clusters[bestcluster].append(x)
        else:
            clusters[bestcluster] = [x]
            
    return clusters
    
    
def reevaluate_centroids(clusters):
    # Calculate the new mean (mu) of each cluster
    mu = [np.mean(clusters[k], axis=0) for k in sorted(clusters.keys())]
        
    return mu
    
    
def has_converged(mu, oldmu):
    # Convergence when the mu and oldmu values are the same
    # We use a set here to make comparison easier
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
    
    
def kmeans(X, k):
    # Initialize to k random centers
    oldmu = random.sample(X, k)
    mu = random.sample(X, k)
    
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_data(X, mu)
        # Reevaluate centers
        mu = reevaluate_centroids(clusters)
        
    return (mu, clusters)
    
    
def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def outliers(clusters, centroids, nsigma=3):
    outliers = []
    
    for label, cluster in clusters.iteritems():
        center = centroids[label]
    
        # Calculate the Euclidian distances from each point to the center of the cluster
        distances_center = {euclidian_distance(x, center): x for x in cluster}
        median = np.median(distances_center.keys())    
        sigma = np.std(distances_center.keys())  
        
        # An outlier is a point thats nsigma away from the median of the Euclidian distances
        # We use the median instead of the mean, because outliers have high impact on the mean
        cluster_outliers = [point for dist,point in distances_center.iteritems() if dist > median+nsigma*sigma or dist < median-nsigma*sigma]
        outliers.extend(cluster_outliers)
        
    return outliers

    
if __name__ == "__main__":
    # Number of clusters we have
    nclusters = 7
    
    # Read the data from the input file
    data = np.genfromtxt('./anomaly.csv', delimiter=',')
    
    # Find the clusters and their centroids
    centroids, clusters = kmeans(data, nclusters)
        
    # Find all points that are 3 sigma (99.7% confidence interval) away from the centroid
    anomalies = outliers(clusters, centroids, nsigma=4)
    

    colors = cycle('bgrcmy')
    
    for label, cluster in clusters.iteritems():
        cluster = np.array(cluster)
        # Plot the points of the cluster
        d = plt.scatter(cluster[:,0], cluster[:,1], c=colors.next())
        # Plot the center of the cluster
        c = plt.scatter(centroids[label][0], centroids[label][1], c='r', marker='x')
        # Plot the anomalies
        anomalies = np.array(anomalies)        
        a= plt.scatter(anomalies[:,0], anomalies[:,1], marker='o', s=80, facecolors='none', edgecolors='r')
        
        
    plt.legend((c, a), ('Centroid', 'Anomaly'), scatterpoints=1, loc='best', fontsize=8)
    plt.show()