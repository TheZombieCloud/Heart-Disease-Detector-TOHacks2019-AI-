import csv
import glob

class EuclideanDistance:
    def __init__(self, distance, classification):
        self._distance = distance
        self._classification = classification
    def getDistance(self):
        return self._distance
    def getClassification(self):
        return self._classification

#default k = 3
def KNearestNeighbours(data, predict, k=3):
    dist = []

datasets = {"0":[], "1": [], "3": [], "4": []}

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
            datasets[data[13]].append(data)



