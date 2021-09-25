import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import StratifiedKFold
from fisher_score import fisher_score, feature_ranking
from reliefF import reliefF
from random_forest import apply_RF
from SVM import apply_SVM_RFE
from Simple_cont import simple_MI
from Iterative_cont import iterative_MI
import random
import matplotlib;
import copy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from classify import predict
from sklearn.metrics import accuracy_score
dataset = 'lung.mat'
mat = scipy.io.loadmat(dataset)
X = mat['X']    # data
X = X.astype(float)
y = mat['Y']    # label
y = y[:, 0]

X2 = pd.DataFrame(X)
X2 = X2.values

y2 = np.array(y)
y2 = np.reshape(y, -1)

cv = StratifiedKFold(n_splits=10, shuffle=True)

scores_fisher = fisher_score(X=X, y=y)
ranking_fisher = feature_ranking(score=scores_fisher)

scores_relief = reliefF(X=X, y=y)
ranking_relief = feature_ranking(score=scores_relief)

scores_random = apply_RF(X=X, y=y)
ranking_random = feature_ranking(score=scores_random)

ranking_svm = apply_SVM_RFE(X=X, y=y)
ranking_svm = ranking_svm[0]

ranking_simple = simple_MI(X=X2, y=y2)

ranking_iter = iterative_MI(X=X2, y=y2)
ranking_iter = ranking_iter['selected_features']


fisher_acc_score = []
relief_acc_score = []
random_acc_score = []
svm_acc_score = []
simple_acc_score = []
iter_acc_score = []



def run(clf, feature_idx, n_features):

    for i, (train, test) in enumerate(cv.split(X, y)):
        predictions = predict(clf,
                              X[train][:, feature_idx[:n_features]],
                              X[test][:, feature_idx[:n_features]],
                              y[train],
                              y[test]
                              )

        if feature_idx is ranking_fisher:
            acc = accuracy_score(predictions, y[test])
            fisher_acc_score.append(acc)
            mean_acc = np.mean(fisher_acc_score, axis=0)
            return mean_acc
        fisher_acc_score.clear()
        if feature_idx is ranking_relief:
            acc = accuracy_score(predictions, y[test])
            relief_acc_score.append(acc)
            mean_acc = np.mean(relief_acc_score, axis=0)
            return mean_acc
        relief_acc_score.clear()
        if feature_idx is ranking_random:
            acc = accuracy_score(predictions, y[test])
            random_acc_score.append(acc)
            mean_acc = np.mean(random_acc_score, axis=0)
            return mean_acc
        random_acc_score.clear()
        if feature_idx is ranking_svm:
            acc = accuracy_score(predictions, y[test])
            svm_acc_score.append(acc)
            mean_acc = np.mean(svm_acc_score, axis=0)
            return mean_acc
        svm_acc_score.clear()
        if feature_idx is ranking_simple:
            acc = accuracy_score(predictions, y[test])
            simple_acc_score.append(acc)
            mean_acc = np.mean(simple_acc_score, axis=0)
            return mean_acc
        simple_acc_score.clear()
        if feature_idx is ranking_iter:
            acc = accuracy_score(predictions, y[test])
            iter_acc_score.append(acc)
            mean_acc = np.mean(iter_acc_score, axis=0)
            return mean_acc



all_acc_score = []

def run_all(clf):
    for i, (train, test) in enumerate(cv.split(X, y)):

        predictions = predict(clf,
                              X[train],
                              X[test],
                              y[train],
                              y[test]
                              )
        acc = accuracy_score(predictions, y[test])
        all_acc_score.append(acc)
    mean_acc = np.mean(all_acc_score, axis=0)
    return mean_acc

accuracy_fisher = []
accuracy_relief = []
accuracy_random = []
accuracy_svm = []
accuracy_simple = []
accuracy_iter = []


def plot_accuracy(clf):


    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_fisher, n)
        accuracy_fisher.append(accuracy)

    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_relief, n)
        accuracy_relief.append(accuracy)

    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_random, n)
        accuracy_random.append(accuracy)

    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_svm, n)
        accuracy_svm.append(accuracy)

    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_simple, n)
        accuracy_simple.append(accuracy)

    for n in range(1, 201, 1):
        accuracy = run(clf, ranking_iter, n)
        accuracy_iter.append(accuracy)

    accuracy_all = run_all(clf)

    # num_features = list(range(1, 101, 1))

    plt.figure()
    plt.style.use('tableau-colorblind10')
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams['savefig.dpi'] = 1200

    # Available style sheets:
    # print(plt.style.available)
    dim = np.arange(1, 201)
    plt.plot(dim, accuracy_fisher, label="Fisher", marker="o", markevery=5, markersize=4)
    plt.plot(dim, accuracy_relief, label="ReliefF", marker="v", markevery=5, markersize=4)
    plt.plot(dim, accuracy_random, label="Random forrest", marker="s", markevery=5, markersize=4)
    plt.plot(dim, accuracy_svm, label="SVM-RFE", marker="x", markevery=5, markersize=4)
    plt.plot(dim, accuracy_simple, label="Simple MI", marker="D", markevery=5, markersize=4)
    plt.plot(dim, accuracy_iter, label="Iterative MI", marker="p", markevery=5, markersize=4)
    plt.axhline(y=accuracy_all, label="All Features Included", linestyle="dashed", color="black")


    plt.legend(loc="best", prop={'size': 8})

    # plt.ylim(0, 1.0)
    plt.xticks([1, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    plt.xlim([1, 201])

    plt.xlabel('Number of features variables')
    plt.ylabel('Classification accuracy')
    plt.ylabel('Classification accuracy')
    print(all_acc_score)
    print(accuracy_all)

    plt.savefig("%s_%s_200.png" % (dataset, clf))
    accuracy_fisher.clear()
    accuracy_relief.clear()
    accuracy_random.clear()
    accuracy_svm.clear()
    accuracy_simple.clear()
    accuracy_iter.clear()
    iter_acc_score.clear()
    all_acc_score.clear()



plot_accuracy(clf='naive-bayes')
plot_accuracy(clf='kNN')
plot_accuracy(clf='logistic-regression')
plot_accuracy(clf='neural-net')
plot_accuracy(clf='random-forest')
plot_accuracy(clf='SVM')
