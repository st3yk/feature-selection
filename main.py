from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import selectors_lib

if __name__ == '__main__':
    print("hello world!")
    iris = load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    data = ["Iris", "Wine", "Breast cancer"]
    X = [iris.data, wine.data, breast_cancer.data]
    y = [iris.target, wine.target, breast_cancer.target]
    
    for i in range(len(data)):
        print("\n" + data[i] + ", " + str(len(X[i][0])) + " features:")
        
        forward_search = selectors.forward_search(X[i], y[i])
        variance_threshold = selectors.variance_threshold(X[i])

        no_selectors = selectors.get_score(X[i], y[i]) 
        forward_selector = selectors.get_score(forward_search, y[i]) 
        variance_selector = selectors.get_score(variance_threshold, y[i])
        
        print("No selectors: " + str(selectors.get_score(X[i], y[i])) + "%")
        print("Forward search selector: " + str(forward_selector) + "%")
        print("Variance threshold selector: " + str(variance_selector) + "%")
