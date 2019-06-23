import csv
import math
'''
medians = [0] * len(train_set_x[0])
for i in train_set_x:
    for j in range(len(i)):
        if i[j] != "q":
            medians[j] += i[j]
for i in range(len(medians)):
    medians[i] /= len(train_set_x)
for i in range(len(train_set_x)):
    for j in range(len(train_set_x[i])):
        if train_set_x[i][j] == "q":
            train_set_x[i][j] = medians[j]
            '''
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
    for i in range (0, 2):
        for line in data[str(i)]:
            euclidean_distance = 0.0
            for point in range (0, len(line)):
                euclidean_distance += math.pow((float(line[point])-float(predict[point])), 2)
            euclidean_distance = math.sqrt(euclidean_distance)
            dist.append(EuclideanDistance(euclidean_distance, i))
    dist.sort(key = lambda x: x.getDistance())
    vote = [0, 0]
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

datasets = {"0": [], "1": []}

with open ("heart.csv") as fileIn:
    reader = csv.reader(fileIn)
    first = True

    counter = 0

    for line in reader:
        if first:
            first = False
            continue
        counter += 1
        if counter<17:
            continue
        data = line
        missing = False
        for i in range(0, len(data)):
            if data[i] == "?":
                missing = True
                break
        if missing:
            continue
        datasets[data[13]].append(data[2:-1])

predict = []
answers = []

with open ("heart.csv") as fileIn:
    reader = csv.reader(fileIn)
    first = True

    counter = 0

    for line in reader:
        if first:
            first = False
            continue
        counter += 1
        if counter == 17:
            break
        data = line
        missing = False
        for i in data:
            if i == "?":
                missing = True
                break
        if missing:
            continue
        predict.append(data[2:-1])
        answers.append(data[-1:])

correct = 0
total = 0

for i in range(0, len(predict)):
    total += 1
    if int(answers[i][0]) == KNearestNeighbours(datasets, predict[i]):
        correct += 1

print ("Confidence: " + str(correct/total))

