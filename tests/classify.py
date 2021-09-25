import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve

classifers = {
    'naive-bayes': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'logistic-regression': LogisticRegression(),
    'neural-net': MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100,100),max_iter=50000),
    'random-forest': RandomForestClassifier(),
    # 'SVM': LinearSVC(dual=False)
    'SVM': SVC(kernel='rbf')

}

def predict(clf, X_train, X_test, y_train, y_test):
    model = classifers[clf]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


