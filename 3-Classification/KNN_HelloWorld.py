# Example of making predictions
from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
       neighbors.append(distances[i][0])
    return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 
# Test distance function
dataset = [[2.7, 2.5, 0],
           [1.4, 2.3, 0],
           [3.3, 4.4, 0],
           [1.3, 1.8, 0],
           [3.0, 3.0, 0],
           [7.6, 2.7, 1],
           [5.3, 2.0, 1],
           [6.9, 1.7, 1],
           [8.6,-0.2, 1],
           [7.6, 3.5, 1]]
#prediction = predict_classification(dataset, dataset[0], 3)
z =  [7.0, 2.5, 1]
prediction = predict_classification(dataset, z, 3)
print('Expected %d, Got %d.' % (z[-1], prediction))