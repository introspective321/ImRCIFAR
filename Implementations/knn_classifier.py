#Calculating Euclidean distance
def euclidean_dist(pointA, pointB):

    distance = np.square(pointA - pointB) # (ai-bi)**2 for every point in the vectors
    distance = np.sum(distance) # adds all values
    distance = np.sqrt(distance)
    return distance

#Calculating distance of test point from all the training points
def distance_from_all_training_points(test_point):
#Returns- dist_array- Array holding distance values for all training data points
    dist_array = np.array([])
    for train_point in X_train_pca:
        dist = euclidean_dist(test_point, train_point)
        dist_array = np.append(dist_array, dist)
    return dist_array

#Implementing KNN Classification model
def KNNClassifier(train_features, train_target, test_features, k = 5):
  predictions = np.array([])
  train_target = train_target.reshape(-1,1)
  for test_point in test_features: # iterating through every test data point
    dist_array = distance_from_all_training_points(test_point).reshape(-1, 1)
    neighbors = np.concatenate((dist_array, train_target), axis=1)
    neighbors_sorted = neighbors[neighbors[:, 0].argsort()] # sorts training points on the basis of distance
    k_neighbors = neighbors_sorted[:k] # selects k-nearest neighbors
    target_class = np.argmax(np.bincount(k_neighbors[:, 1].astype(int)))# selects label with highest frequency
    predictions = np.append(predictions, target_class)
  return predictions

#running inference on test data
test_predictions = KNNClassifier(X_train_pca, y_training, X_test_pca, k = 5)
test_predictions

from sklearn.metrics import classification_report
#Model Evaluation
def accuracy(y_test, y_preds):
    total_correct = sum(1 for true, pred in zip(y_test, y_preds) if true == pred)
    acc = total_correct / len(y_test)
    return acc
acc = accuracy(y_testing, test_predictions)
print('Model accuracy (Scratch) = ', acc*100)
print("Score:\n", classification_report(y_testing, test_predictions))

import seaborn as sns
k_values = list(range(1,20))
accuracy_list = []
for k in k_values:
    test_predictions = KNNClassifier(X_train_pca, y_training, X_test_pca, k)
    accuracy_list.append(accuracy(y_testing, test_predictions))
sns.barplot(k_values, accuracy_list)