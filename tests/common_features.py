import numpy as np
from numpy import array_equal
import pandas as pd
from scipy import io
from sklearn.model_selection import StratifiedKFold
from fisher_score import fisher_score
from reliefF import reliefF
from random_forest import apply_RF
from SVM import apply_SVM_RFE
from Simple_cont import simple_MI
from Iterative_cont import iterative_MI
from itertools import cycle

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom

from classify import predict
from sklearn.metrics import accuracy_score

# Load Data
dataset = 'colon.mat'
in_file = io.loadmat(dataset)
X = pd.DataFrame(in_file['X'], dtype=float)
y = pd.DataFrame(in_file['Y'])

# convert classes from whatever datatype they are to binary integers (0 and 1)
y_values = np.unique(y)

y_binary = np.array(y == y_values[0], dtype=int)

X = X.values

y = np.reshape(y_binary, -1)

cv = StratifiedKFold(n_splits=10, shuffle=True)


def feature_ranking(score):
    """
    Rank features in descending order according to fisher score, the larger the fisher score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]


scores_fisher = fisher_score(X=X, y=y)
ranking_fisher = feature_ranking(score=scores_fisher)

scores_relief = reliefF(X=X, y=y)
ranking_relief = feature_ranking(score=scores_relief)

scores_random = apply_RF(X=X, y=y)
ranking_random = feature_ranking(score=scores_random)

ranking_svm = apply_SVM_RFE(X=X, y=y)
ranking_svm = ranking_svm[0]

ranking_simple = simple_MI(X=X, y=y)

ranking_iter = iterative_MI(X=X, y=y)
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


# Enter high performing FS methods to find common features between them
best_methods = [ranking_fisher, ranking_simple, ranking_iter]

# Function to test if array is in list of arrays
def arreq_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if array_equal(elem, myarr)), False)


# Get common features between best methods
def common(num_comm):
    feature_subsets = []
    for i in best_methods:
        feature_subsets.append(i[:num_comm])
    common_subset = set.intersection(*map(set, feature_subsets))
    ranking_common = np.array(list(common_subset))
    num_features = len(ranking_common)
    return ranking_common, num_features


common_acc_score = []


def run_common(clf, feature_idx):
    for i, (train, test) in enumerate(cv.split(X, y)):
        predictions = predict(clf,
                              X[train][:, feature_idx],
                              X[test][:, feature_idx],
                              y[train],
                              y[test]
                              )

        acc = accuracy_score(predictions, y[test])
        common_acc_score.append(acc)
        mean_acc = np.mean(common_acc_score, axis=0)
        return mean_acc
    common_acc_score.clear()


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


# binomial ppf sig line
unique, counts = np.unique(y, return_counts=True)
class_bal = counts[1] / len(y)
n, p = len(X[1]), class_bal

ppf_val = binom.ppf(0.05, n, p)
sig_line = ppf_val / len(X[1])

accuracy_common = []
num_common_features = []
accuracy_fisher = []
accuracy_relief = []
accuracy_random = []
accuracy_svm = []
accuracy_simple = []
accuracy_iter = []
methods_used_labels = []
accuracy_methods = []


def plot_accuracy(clf):
    # common features
    for n in range(10, 101, 5):
        ranking_common, num_features = common(n)
        accuracy = run_common(clf, ranking_common)
        accuracy_common.append(accuracy)
        num_common_features.append(num_features)

    # all features
    accuracy_all = run_all(clf)

    # fs methods
    if arreq_in_list(ranking_fisher, best_methods):
        methods_used_labels.append("Fisher")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_fisher, n)
            accuracy_fisher.append(accuracy)
        accuracy_methods.append(accuracy_fisher)

    if arreq_in_list(ranking_relief, best_methods):
        methods_used_labels.append("ReliefF")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_relief, n)
            accuracy_relief.append(accuracy)
        accuracy_methods.append(accuracy_relief)

    if arreq_in_list(ranking_random, best_methods):
        methods_used_labels.append("Random Forest")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_random, n)
            accuracy_random.append(accuracy)
        accuracy_methods.append(accuracy_random)

    if arreq_in_list(ranking_svm, best_methods):
        methods_used_labels.append("SVM-RFE")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_svm, n)
            accuracy_svm.append(accuracy)
        accuracy_methods.append(accuracy_svm)

    if arreq_in_list(ranking_simple, best_methods):
        methods_used_labels.append("Simple MI")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_simple, n)
            accuracy_simple.append(accuracy)
        accuracy_methods.append(accuracy_simple)

    if arreq_in_list(ranking_iter, best_methods):
        methods_used_labels.append("Iterative MI")
        for n in range(10, 101, 5):
            accuracy = run(clf, ranking_iter, n)
            accuracy_iter.append(accuracy)
        accuracy_methods.append(accuracy_iter)

    fig, ax = plt.subplots()

    plt.style.use('tableau-colorblind10')
    plt.rcParams["figure.facecolor"] = "w"
    # plt.rcParams['savefig.dpi'] = 1200

    dim = range(10, 101, 5)
    # Plot common features accuracy, all features, and sig. line
    plt.plot(dim, accuracy_common, label="Common Features", marker="x", markersize=4)

    plt.axhline(y=accuracy_all, label="All Features Included", linestyle="dashed", color="black")
    plt.axhline(y=sig_line, label="Significance p=0.05", linestyle="dashed", color="#C85200")

    colours = cycle(["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200", "#898989", "#A2C8EC"])

    # plot individual FS methods used
    for accuracy_method, label in zip(accuracy_methods, methods_used_labels):
        plt.plot(dim, accuracy_method, label=label, color=next(colours))

    plt.legend(loc="best", prop={'size': 8})

    ax.set_xticks(range(10, 101, 10))
    ax.set_xlim([9, 101])

    sec_labels = num_common_features[0::2]
    ax2 = ax.twiny()
    ax2.set_xticks(range(10, 101, 10))
    ax2.set_xticklabels(sec_labels)
    ax2.set_xlim([9, 101])
    ax2.set_xlabel("Number of common features")

    ax.set_xlabel('Feature subset size')
    ax.set_ylabel('Classification accuracy')

    # Text box
    max_acc = max(accuracy_common)
    max_index = accuracy_common.index(max_acc)
    best_common = num_common_features[max_index]
    subset_sizes = range(10, 101, 5)
    best_subset = subset_sizes[max_index]

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0, -0.22,
            "Max. accuracy=%.2f for %s common features (feature subset size=%s)\nSelection methods used: %s"
            % (max_acc, best_common, best_subset, ', '.join(methods_used_labels)),
            transform=ax.transAxes, fontsize=8, bbox=props)
    fig.subplots_adjust(bottom=0.2)

    plt.savefig("common_%s_%s.png" % (dataset, clf))

    accuracy_common.clear()
    all_acc_score.clear()
    num_common_features.clear()
    accuracy_fisher.clear()
    accuracy_relief.clear()
    accuracy_random.clear()
    accuracy_svm.clear()
    accuracy_simple.clear()
    accuracy_iter.clear()
    iter_acc_score.clear()
    methods_used_labels.clear()
    accuracy_methods.clear()


plot_accuracy(clf='naive-bayes')
plot_accuracy(clf='kNN')
plot_accuracy(clf='logistic-regression')
plot_accuracy(clf='neural-net')
plot_accuracy(clf='random-forest')
plot_accuracy(clf='SVM')
