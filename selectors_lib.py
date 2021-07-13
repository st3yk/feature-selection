import numpy as np
from random import randrange 
from sklearn.neighbors import KNeighborsClassifier 

def randomize_data(X, y):
    size = len(X)
    new_X = X.tolist()
    new_y = y.tolist()
    for _ in range(1000):
        index1 = randrange(len(X))
        index2 = randrange(len(X))
        new_X[index1], new_X[index2] = new_X[index2], new_X[index1]
        new_y[index1], new_y[index2] = new_y[index2], new_y[index1]
    return new_X, new_y

def split_data(X, y):
    split = int(len(X) / 3)
    X_train = X[split:] 
    y_train = y[split:] 
    X_test = X[:split]
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
    return round((score / len(predicted)) * 100, 2)

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
            XX = np.array(X)[:, curr].tolist()
            score = knn_score(XX, y)
            if score > curr_best_score:
                curr_best_score = score
                curr_best = curr[:]
            
            if score > best_score:
                best_score = score
                best = curr[:]
            curr.remove(num)
    new_X = np.array(X)[:, best].tolist()
    return new_X
    
