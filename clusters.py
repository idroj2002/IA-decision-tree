from typing import Tuple, List
from math import sqrt
import sys


def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    distance = sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return 1 / (1 + distance)


def euclidean_squared(v1, v2):
    return euclidean(v1, v2) ** 2


def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v ** 2 for v in v1])
    sum2sq = sum([v ** 2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1 ** 2 / len(v1)) * (sum2sq - sum2 ** 2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist


def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    current_clust_id = -1  # Non original clusters have negative id
    centroids = []
    distances_sum = 0.0

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    while len(clust) > 1:  # Termination criterion
        lowest_pair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i + 1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)

                d = distances[(clust[i].id, clust[j].id)]

                # update closest and lowest_pair if needed
                if d < closest:
                    closest = d
                    lowest_pair = (i, j)

        # Calculate the average vector of the two clusters
        merge_vec = [(clust[lowest_pair[0]].vec[i] + clust[lowest_pair[1]].vec[i]) / 2.0
                     for i in range(len(clust[0].vec))]

        # Create the new cluster
        new_cluster = BiCluster(merge_vec, left=clust[lowest_pair[0]], right=clust[lowest_pair[1]], dist=closest,
                                id=current_clust_id)
        centroids.append(merge_vec)
        distances_sum += closest

        # Update the clusters
        current_clust_id -= 1
        del clust[lowest_pair[1]]
        del clust[lowest_pair[0]]
        clust.append(new_cluster)

    return clust[0], centroids, distances_sum


def print_clust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels is None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left is not None:
        print_clust(clust.left, labels=labels, n=n + 1)
    if clust.right is not None:
        print_clust(clust.right, labels=labels, n=n + 1)


# ......... K-MEANS ..........
def kcluster(rows,distance=pearson,k=4):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),
    max([row[i] for row in rows])) for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters=[[random.random()*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]

    last_matches = None
    for t in range(100):
        best_matches = [[] for i in range(k)]

        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            best_match = 0
            for i in range(k):
                d = distance(clusters[i], row)
            if d < distance(clusters[best_match], row): best_match = i
            best_matches[best_match].append(j)

        # If the results are the same as last time, done
        if best_matches == last_matches: break
        last_matches = best_matches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs = [0.0] * len(rows[0])
            if len(best_matches[i]) > 0:
                for rowid in best_matches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m] += rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(best_matches[i])
            clusters[i] = avgs
    return best_matches

class KMeans:
    def __init__(self, rows, distance=euclidean_squared, k=4):
        self.rows = rows
        self.distance = distance
        self.k = k
        self.clusters = None

    def start_configuration(self, iterations=10, bestresult=None):
        for _ in range(iterations):
            initial_centroids = self.centroids_inicialization()
            centroids, totaldistance = self.assign_cluster(initial_centroids)  # Centroids and total distance

            if bestresult is None or totaldistance > bestresult[
                1]:  # If best result is None or total distance is greater than best result
                bestresult = (centroids, totaldistance)  # Update best result

        return bestresult

    def centroids_inicialization(self):  # Random centroids
        ranges = [(min([row[i] for row in self.rows]),
                   max([row[i] for row in self.rows])) for i in range(len(self.rows[0]))]  # Get ranges for each column

        centroids = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                      for i in range(len(self.rows[0]))] for j in range(self.k)]  # Get random centroids

        return centroids

    def assign_cluster(self, centroids, iterations=100, lastmatches=None):
        for _ in range(iterations):
            # print("Iteració %i" % t)
            # print(centroids[0][0])
            distances, matches = self.best_centroid(centroids)

            if matches == lastmatches:  # If matches are the same as last matches stop
                break
            lastmatches = matches

            self.update_centroids(centroids, matches)  # update centroids

        self.clusters = matches
        return (centroids, sum(distances))

    def best_centroid(self, centroids):  # Get best centroid for each row
        bestmatches = [[] for i in range(len(centroids))]
        bestdistances = [0 for i in range(len(self.rows))]

        for idrow, row in enumerate(self.rows):
            bestmatch = 0
            bestdistance = self.distance(centroids[0], row)

            # Find the closest centroid
            for centroid_id, centroid in enumerate(centroids):
                distance = self.distance(centroid, row)
                if distance > self.distance(centroids[bestmatch], row):
                    bestmatch = centroid_id
                    bestdistance = distance

            # Store results
            bestdistances[idrow] = bestdistance
            bestmatches[bestmatch].append(idrow)

        return (bestdistances, bestmatches)

    def update_centroids(self, centroids, matches):
        # For each centroid
        for centroid, _ in enumerate(centroids):
            avgs = [0.0] * len(self.rows[0])

            if len(matches[centroid]) > 0:
                # For each item in the cluster
                for rowid in matches[centroid]:
                    # Add each value to the average
                    for attrid in range(len(self.rows[rowid])):
                        avgs[attrid] += self.rows[rowid][attrid]

                # Divide by number of items to get the average
                for j in range(len(avgs)):
                    avgs[j] /= len(matches[centroid])

                    # Update the centroid
                    centroids[centroid] = avgs


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "blogdata_full.txt"

    row_names, headers, data = readfile(filename)
    cluster, centroids, distances_sum = hcluster(data)
    print_clust(cluster, row_names)
    print(centroids)
    print(distances_sum)


if __name__ == "__main__":
    main()
