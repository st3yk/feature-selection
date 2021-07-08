import numpy as np
from random import randrange 
from sklearn.neighbors import KNeighborsClassifier 

def randomize_data(X, y):
    for _ in range(1000):
        index1 = randrange(len(X)) 
        index2 = randrange(len(X)) 
        X[index1], X[index2] = X[index2], X[index1]
        y[index1], y[index2] = y[index2], y[index1]
    return X, y

def split_data(X, y, train=0.2):
    split = int(len(X) * train)
    X_train = X[split:]
    X_test = X[:split]
    y_train = y[split:]
    y_test = y[:split]
    return X_train, X_test, y_train, y_test

def knn_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    score = 0
    for i in range(len(predicted)):
        if predicted[i] == y_test[i]:
            score += 1
    return (score / len(y_test)) * 100

def variance_threshold(X, threshold=0):
    new_X = np.copy(X)
    variances = np.var(new_X, axis=0)
    drop = []
    for i in range(len(variances)):
        if variances[i] <= threshold:
            drop.append(i)
    for i in range(len(drop)):
        new_X = np.delete(new_X, drop[i] - i, 1)
    return new_X
 
def forward_search(X, y, features_number=-1):
    if features_number == -1:
        features_number = np.size(X, axis=1)
    length = np.size(X, axis = 0)
    best = []
    best_score = 0
    curr_best = []
    for i in range(features_number):
        curr = curr_best[:]
        curr_best_score = 0
        numbers = list(range(features_number))
        for n in curr:
            numbers.remove(n)
        for num in numbers:
            curr.append(num)
            curr.sort()
            XX = X[:, curr]
            score = knn_score(XX, y)
            if score > curr_best_score:
                curr_best_score = score
                curr_best = curr[:]
            
            if score > best_score:
                best_score = score
                best = curr[:]
            curr.remove(num)
    new_X = X[:, best]
    return new_X
    
