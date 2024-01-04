from typing import Tuple, List
from math import sqrt
import sys
import random
import matplotlib.pyplot as plt


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
def kcluster(rows, distance=euclidean_squared, k=4):  # function from lectures
    # Determine the minimum and maximum values for each point
    ranges = [(min([row[i] for row in rows]),
               max([row[i] for row in rows])) for i in range(len(rows[0]))]

    # Create k randomly placed centroids
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(rows[0]))] for j in range(k)]

    last_matches = None
    best_matches = [[] for i in range(k)]
    best_distances = [0 for _ in range(len(rows))]
    for t in range(100):
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row = rows[j]
            best_match = 0
            best_distance = 0.0
            for i in range(k):
                d = distance(clusters[i], row)
                if d < distance(clusters[best_match], row):
                    best_match = i
                    best_distance = d
            best_matches[best_match].append(j)
            best_distances[j] = best_distance

        # If the results are the same as last time, done
        if best_matches == last_matches:
            break
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
    return best_matches, sum(best_distances)


class KMeans:
    def __init__(self, rows, distance=euclidean_squared, k=4):
        self.rows = rows
        self.distance = distance
        self.k = k
        self.clusters = None

    def run_n_iter(self, iterations=10, best_result=None):
        for _ in range(iterations):
            initial_centroids = self.centroids_init()
            centroids, total_distance = self.build_cluster(initial_centroids)  # Centroids and total distance

            if best_result is None or total_distance < best_result[1]:  # If best result is None or total distance is
                # greater than best result
                best_result = (centroids, total_distance)  # Update best result

        return best_result

    def centroids_init(self):  # Random centroids
        ranges = [(min([row[i] for row in self.rows]),
                   max([row[i] for row in self.rows])) for i in range(len(self.rows[0]))]  # Get ranges for each column

        centroids = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                      for i in range(len(self.rows[0]))] for _ in range(self.k)]  # Get random centroids

        return centroids

    def build_cluster(self, centroids, iterations=100, last_matches=None):
        matches = None
        distances = None

        for _ in range(iterations):
            # print("Iteració %i" % t)
            # print(centroids[0][0])
            distances, matches = self.best_centroids(centroids)

            if matches == last_matches:  # If matches are the same as last matches stop
                break
            last_matches = matches

            self.update_centroids(centroids, matches)  # update centroids

        self.clusters = matches
        return centroids, sum(distances)

    def best_centroids(self, centroids):  # Find which centroid is the closest for each row
        best_matches = [[] for _ in range(self.k)]
        best_distances = [0 for _ in range(len(self.rows))]

        for id_row, row in enumerate(self.rows):
            best_match = 0
            best_distance = self.distance(centroids[0], row)

            # Find the closest centroid
            for centroid_id, centroid in enumerate(centroids):
                distance = self.distance(centroid, row)
                if distance > self.distance(centroids[best_match], row):
                    best_match = centroid_id
                    best_distance = distance

            # Store results
            best_distances[id_row] = best_distance
            best_matches[best_match].append(id_row)

        return best_distances, best_matches

    def update_centroids(self, centroids, matches):
        # For each centroid
        for centroid, _ in enumerate(centroids):
            avgs = [0.0] * len(self.rows[0])

            if len(matches[centroid]) > 0:
                # For each item in the cluster
                for row_id in matches[centroid]:
                    # Add each value to the average
                    for attr_id in range(len(self.rows[row_id])):
                        avgs[attr_id] += self.rows[row_id][attr_id]

                # Divide by number of items to get the average
                for j in range(len(avgs)):
                    avgs[j] /= len(matches[centroid])

                # Update the centroid
                centroids[centroid] = avgs


def elbow(data, begin, end, incr, restarts):  # Elbow method to find the best k
    total_distances = []
    for i in range(begin, end, incr):
        kmeans = KMeans(data, k=i)
        _, total_distance = kmeans.run_n_iter(iterations=restarts)
        total_distances.append(total_distance ** 2)
    return total_distances


def plot_elbow(data, begin, end, incr, restarts):  # Plot elbow method
    total_distances = elbow(data, begin, end, incr, restarts)
    plt.plot(range(begin, end, incr), total_distances)
    plt.show()


def main():
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = "blogdata_full.txt"

    row_names, headers, data = readfile(filename)
    kmeans = KMeans(data)
    initial_centroids = kmeans.centroids_init()
    """print("Initial centroids:")
    for centroid in initial_centroids:
        print(centroid)
    print()"""
    centroids, total_distance = kmeans.build_cluster(initial_centroids)
    """print("Final centroids: ")
    for centroid in centroids:
        print(centroid)
    print("Total distance: %f" % totaldistance)
    print()
    """

    print("Total distance: %f" % total_distance)

    restarts = 10
    kmeans = KMeans(data)
    centroids, total_distance = kmeans.run_n_iter(iterations=restarts)
    print("%i Restarts -> Distance to centroids: %2.3f\n"
          % (restarts, total_distance))

    result = elbow(data, 1, 10, 1, restarts)
    print("Elbow method results: " + str(result))

    for i in range(len(result)):
        print("%i Clusters - %i Restarts -> Total Distance is: %2.3f"
              % (i + 1, restarts, result[i]))

    plot_elbow(data, 1, 10, 1, restarts)


if __name__ == "__main__":
    main()
