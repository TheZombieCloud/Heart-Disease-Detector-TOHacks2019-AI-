import csv
import math
import glob

class EuclideanDistance:
    def __init__(self, distance, classification):
        self._distance = distance
        self._classification = classification
    def getDistance(self):
        return self._distance
    def getClassification(self):
        return self._classification

#default k = 13
def KNearestNeighbours(data, predict, k=13):
    dist = []
    for i in range (0, 5):
        for line in data[str(i)]:
            euclidean_distance = 0.0
            for point in range (0, len(line)):
                euclidean_distance += math.pow((line[point]-predict[point]), 2)
            euclidean_distance = math.sqrt(euclidean_distance)
            dist.append[EuclideanDistance(euclidean_distance, i)]
    dist.sort(key = lambda x: x.getDistance())
    vote = [0, 0, 0, 0, 0]
    for i in range (0, k):
        decision = dist[i].getClassification()
        vote[int(decision)] += 1
    max = 0
    cur = -1
    for i in range (0, len(vote)):
        if vote[i] > max:
            max = vote[i]
            cur = i
    return cur

datasets = {"0": [], "1": [], "2": [], "3": [], "4": []}

for file in glob.glob("processed.*.data.csv"):
    with open (file) as fileIn:
        reader = csv.reader(fileIn)

        for line in reader:
            data = line
            missing = False
            for i in data:
                if i == "?":
                    missing = True
                    break
            if missing:
                continue
            datasets[data[13]].append(data[:-1])

predict = []

KNearestNeighbours(datasets, predict)

