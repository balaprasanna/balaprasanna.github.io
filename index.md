---
layout: home_layout
---

[Link to another page](another-page).
### [](#header-2)What is k-Nearest Neighbors

The model for kNN is the entire training dataset. When a prediction is required for a unseen data instance, the kNN algorithm will search through the training dataset for the k-most similar instances. The prediction attribute of the most similar instances is summarized and returned as the prediction for the unseen instance.

The similarity measure is dependent on the type of data. For real-valued data, the Euclidean distance can be used. Other other types of data such as categorical or binary data, Hamming distance can be used.

In the case of regression problems, the average of the predicted attribute may be returned. In the case of classification, the most prevalent class may be returned.


### [](#header-3)KNN Simple Implementation non vectorized
```python
# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
  with open(filename, 'rb') as csvfile:
      lines = csv.reader(csvfile)
      dataset = list(lines)
      for x in range(len(dataset)-1):
          for y in range(4):
              dataset[x][y] = float(dataset[x][y])
          if random.random() < split:
              trainingSet.append(dataset[x])
          else:
              testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
  distance = 0
  for x in range(length):
    distance += pow((instance1[x] - instance2[x]), 2)
  return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
  distances = []
  length = len(testInstance)-1
  for x in range(len(trainingSet)):
    dist = euclideanDistance(testInstance, trainingSet[x], length)
    distances.append((trainingSet[x], dist))
  distances.sort(key=operator.itemgetter(1))
  neighbors = []
  for x in range(k):
    neighbors.append(distances[x][0])
  return neighbors

def getResponse(neighbors):
  classVotes = {}
  for x in range(len(neighbors)):
    response = neighbors[x][-1]
    if response in classVotes:
      classVotes[response] += 1
    else:
      classVotes[response] = 1
  sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
  return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
  correct = 0
  for x in range(len(testSet)):
    if testSet[x][-1] == predictions[x]:
      correct += 1
  return (correct/float(len(testSet))) * 100.0
  
def main():
  # prepare data
  trainingSet=[]
  testSet=[]
  split = 0.67
  loadDataset('iris.data', split, trainingSet, testSet)
  print 'Train set: ' + repr(len(trainingSet))
  print 'Test set: ' + repr(len(testSet))
  # generate predictions
  predictions=[]
  k = 3
  for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    result = getResponse(neighbors)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
  accuracy = getAccuracy(testSet, predictions)
  print('Accuracy: ' + repr(accuracy) + '%')
  
main()

```

