import json
import pickle
import numpy as np
import os

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from numpyencoder import NumpyEncoder

from preprocessing import preprocess, apply_pca, apply_lda, apply_umap
from visualization import plot_data

# Declare the global constants
# Maximum number of iterations for the MLP models
MAX_ITER = 100
# Random state for UMAP to ensure reproducibility
RANDOM_STATE = 42
# Classification options
CLASSIFICATIONS = ["binary", "category", "subcategory"]
# Accuracy metrics
ACCURACY_METRICS = ["accuracy", "precision", "recall", "f1"]
# Compressions
REDUCERS = ["original", "pca8", "pca6", "pca3", "lda8", "lda6", "lda3", "umap8", "umap6", "umap3"]
# Performance metrics
PERFORMANCE_METRICS = ["fit_time", "prediction_time"]
# Metric options for multiple classification
OPTIONS = ["micro", "macro", "weighted"]

# Prepare the tested Models
MODELS = [
    ("LR", "Logistic Regression", LogisticRegression(solver="sag", max_iter=MAX_ITER)),
    ("SVM", "Support Vector Machine", LinearSVC(penalty="l2", loss="hinge", max_iter=MAX_ITER)),
    ("CART", "Decision Tree", DecisionTreeClassifier(splitter="random")),
    ("MLP5", "Neural Network (One Hidden Layer with 5 Nodes)",
     MLPClassifier((5,), solver="adam", max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ("MLP10", "Neural Network (One Hidden Layer with 10 Nodes)",
     MLPClassifier((10,), solver="adam", max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ("MLP40", "Neural Network (One Hidden Layer with 40 Nodes)",
     MLPClassifier((40,), solver="adam", max_iter=MAX_ITER, shuffle=True, early_stopping=True)),
    ("MLP100", "Neural Network (One Hidden Layer with 100 Nodes)",
     MLPClassifier((100,), solver="adam", max_iter=MAX_ITER, shuffle=True, early_stopping=True))
]

# The cross validation method
KFOLD = model_selection.KFold(n_splits=5, shuffle=True)


# Capitalize a string
def capitalize(_string):
    return _string[0].upper() + _string[1:]


# A function that compares the accuracy of the different neural networks
def compare_algorithms_accuracy(_metrics, _reducer, _metric, _algorithms, _legends, _title):

    # int empty numpy array
    data = np.zeros((4, 3, 5))

    # Fill the data array
    for i in range(4):
        for j in range(3):
            if _metric == "accuracy" or CLASSIFICATIONS[j] == "binary":
                data[i][j] = _metrics[CLASSIFICATIONS[j]][_algorithms[i]][_reducer][_metric]
            else:
                data[i][j] = _metrics[CLASSIFICATIONS[j]][_algorithms[i]][_reducer][_metric]["weighted"]

    # Convert to percentage
    data = data * 100

    # Set the filename
    if _algorithms[0].startswith("MLP"):
        filename = "./figures/MLP" + "_" + _reducer + "_" + _metric
    else:
        filename = "./figures/Compare" + "_" + _reducer + "_" + _metric

    # Set the y label
    if _metric == "accuracy":
        ylabel = capitalize("Accuracy")
    else:
        ylabel = capitalize(_metric) + " (Weighted for Multiple Classification)"

    # Set the x label
    xlabel = "Classification Type"

    # Set the labels of the x-axis per group
    xlabels = ["Binary", "Category", "Subcategory"]

    # Call the plot function
    plot_data(data, xlabel, ylabel, xlabels, _legends, _title, filename)


# A function that compares the performance of the different neural networks
def compare_algorithms_performance(_metrics, _reducer, _metric, _algorithms, _legends, _title):

    # int empty numpy array
    data = np.zeros((4, 3, 5))

    # Fill the data array
    for i in range(4):
        for j in range(3):
            data[i][j] = _metrics[CLASSIFICATIONS[j]][_algorithms[i]][_reducer][_metric]

    # Convert to milliseconds
    data = data * 1000

    # Set the filename
    if _algorithms[0].startswith("MLP"):
        filename = "./figures/MLP" + "_" + _reducer + "_" + _metric
    else:
        filename = "./figures/Compare" + "_" + _reducer + "_" + _metric

    # Set the y label
    if _metric == "fit_time":
        ylabel = "Fit Time (milliseconds)"
    else:
        ylabel = "Prediction Time (milliseconds)"

    # Set the x label
    xlabel = "Classification Type"

    # Set the labels of the x-axis per group
    xlabels = ["Binary", "Category", "Subcategory"]

    # Call the plot function
    plot_data(data, xlabel, ylabel, xlabels, _legends, title, filename, percentage=False)


# A function that compares the accuracy of the algorithms for original and reduced data
def compare_accuracy_dimension(_metrics, _classification, _algorithm, _fullname, _metric, _option=""):

    # int empty numpy array
    data = np.zeros((4, 3, 5))

    # Fill the data array
    if _option:
        data[0][0] = _metrics[_classification][_algorithm]["original"][_metric][_option]
        data[0][1] = _metrics[_classification][_algorithm]["original"][_metric][_option]
        data[0][2] = _metrics[_classification][_algorithm]["original"][_metric][_option]
        data[1][0] = _metrics[_classification][_algorithm]["pca8"][_metric][_option]
        data[1][1] = _metrics[_classification][_algorithm]["lda8"][_metric][_option]
        data[1][2] = _metrics[_classification][_algorithm]["umap8"][_metric][_option]
        data[2][0] = _metrics[_classification][_algorithm]["pca6"][_metric][_option]
        data[2][1] = _metrics[_classification][_algorithm]["lda6"][_metric][_option]
        data[2][2] = _metrics[_classification][_algorithm]["umap6"][_metric][_option]
        data[3][0] = _metrics[_classification][_algorithm]["pca3"][_metric][_option]
        data[3][1] = _metrics[_classification][_algorithm]["lda3"][_metric][_option]
        data[3][2] = _metrics[_classification][_algorithm]["umap3"][_metric][_option]
        # Set the filename
        filename = "./figures/" + _classification + "_" + _algorithm + "_" + _metric + "_" + _option
        # Set the y label
        ylabel = capitalize(_option) + " " + capitalize(_metric)
    else:
        data[0][0] = _metrics[_classification][_algorithm]["original"][_metric]
        data[0][1] = _metrics[_classification][_algorithm]["original"][_metric]
        data[0][2] = _metrics[_classification][_algorithm]["original"][_metric]
        data[1][0] = _metrics[_classification][_algorithm]["pca8"][_metric]
        data[1][1] = _metrics[_classification][_algorithm]["lda8"][_metric]
        data[1][2] = _metrics[_classification][_algorithm]["umap8"][_metric]
        data[2][0] = _metrics[_classification][_algorithm]["pca6"][_metric]
        data[2][1] = _metrics[_classification][_algorithm]["lda6"][_metric]
        data[2][2] = _metrics[_classification][_algorithm]["umap6"][_metric]
        data[3][0] = _metrics[_classification][_algorithm]["pca3"][_metric]
        data[3][1] = _metrics[_classification][_algorithm]["lda3"][_metric]
        data[3][2] = _metrics[_classification][_algorithm]["umap3"][_metric]
        # Set the filename
        filename = "./figures/" + _classification + "_" + _algorithm + "_" + _metric
        # Set the y label
        ylabel = capitalize(_metric)

    # Convert to percentage
    data = data * 100

    # Set the x label
    xlabel = "Dimensionality Reduction Method"

    # Set the labels of the x-axis per group
    xlabels = ["PCA", "LDA", "UMAP"]

    # Set the legends of the figure
    _legends = ["Original", "8 Dimensions", "6 Dimensions", "3 Dimensions"]

    # Set the title
    _title = "Dimensionality Reduction Comparison for " + capitalize(_classification)\
             + " Classification with a " + _fullname

    # Call the plot function
    plot_data(data, xlabel, ylabel, xlabels, _legends, _title, filename)


# A function that compares the performance of the algorithms for original and reduced data
def compare_performance_dimension(_metrics, _classification, _algorithm, _fullname, _metric, _option=""):

    # int empty numpy array
    data = np.zeros((4, 3, 5))

    # Fill the data array
    data[0][0] = _metrics[_classification][_algorithm]["original"][_metric]
    data[0][1] = _metrics[_classification][_algorithm]["original"][_metric]
    data[0][2] = _metrics[_classification][_algorithm]["original"][_metric]
    data[1][0] = _metrics[_classification][_algorithm]["pca8"][_metric]
    data[1][1] = _metrics[_classification][_algorithm]["lda8"][_metric]
    data[1][2] = _metrics[_classification][_algorithm]["umap8"][_metric]
    data[2][0] = _metrics[_classification][_algorithm]["pca6"][_metric]
    data[2][1] = _metrics[_classification][_algorithm]["lda6"][_metric]
    data[2][2] = _metrics[_classification][_algorithm]["umap6"][_metric]
    data[3][0] = _metrics[_classification][_algorithm]["pca3"][_metric]
    data[3][1] = _metrics[_classification][_algorithm]["lda3"][_metric]
    data[3][2] = _metrics[_classification][_algorithm]["umap3"][_metric]

    # Convert to milliseconds
    data = data * 1000

    # Set the filename
    filename = "./figures/" + _classification + "_" + _algorithm + "_" + _metric

    # Set the y label
    if _metric == "fit_time":
        ylabel = "Fit Time (milliseconds)"
    else:
        ylabel = "Predict Time (milliseconds)"

    # Set the x label
    xlabel = "Dimensionality Reduction Method"

    # Set the labels of the x-axis per group
    xlabels = ["PCA", "LDA", "UMAP"]

    # Set the legends of the figure
    _legends = ["Original", "8 Dimensions", "6 Dimensions", "3 Dimensions"]

    # Set the title
    _title = "Dimensionality Reduction Comparison for " + capitalize(_classification) \
             + " Classification with a " + _fullname

    # Call the plot function
    plot_data(data, xlabel, ylabel, xlabels, _legends, _title, filename, percentage=False)


# A function to evaluate performance for each test
def evaluate(_classification, _reducer, _features, _labels):
    if _classification == "binary":
        scoring = ("accuracy", "precision", "recall", "f1")
    else:
        scoring = (
            "accuracy", "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro",
            "recall_weighted", "f1_micro", "f1_macro", "f1_weighted"
        )
    for name, _fullname, model in MODELS:
        print(_classification, ", Reducer : ", _reducer, ", Algorithm : ", _fullname, sep="")
        if len(metrics[_classification][name][_reducer]["accuracy"]):
            print("Skip ... data already available")
        else:
            print("Work in progress ... ", end="")
            results = model_selection.cross_validate(model, _features, _labels, cv=KFOLD, scoring=scoring)
            metrics[_classification][name][_reducer]["fit_time"] = results["fit_time"]
            metrics[_classification][name][_reducer]["prediction_time"] = results["score_time"]
            metrics[_classification][name][_reducer]["accuracy"] = results["test_accuracy"]
            if _classification == "binary":
                metrics[_classification][name][_reducer]["precision"] = results["test_precision"]
                metrics[_classification][name][_reducer]["recall"] = results["test_recall"]
                metrics[_classification][name][_reducer]["f1"] = results["test_f1"]
            else:
                metrics[_classification][name][_reducer]["precision"]["micro"] = results["test_precision_micro"]
                metrics[_classification][name][_reducer]["precision"]["macro"] = results["test_precision_macro"]
                metrics[_classification][name][_reducer]["precision"]["weighted"] = results["test_precision_weighted"]
                metrics[_classification][name][_reducer]["recall"]["micro"] = results["test_recall_micro"]
                metrics[_classification][name][_reducer]["recall"]["macro"] = results["test_recall_macro"]
                metrics[_classification][name][_reducer]["recall"]["weighted"] = results["test_recall_weighted"]
                metrics[_classification][name][_reducer]["f1"]["micro"] = results["test_f1_micro"]
                metrics[_classification][name][_reducer]["f1"]["macro"] = results["test_f1_macro"]
                metrics[_classification][name][_reducer]["f1"]["weighted"] = results["test_f1_weighted"]
            print("Done")
            writefile = open("results/metrics.pkl", "wb")
            pickle.dump(metrics, writefile)
            writefile.close()
        accuracy = round(metrics[_classification][name][_reducer]["accuracy"].mean() * 100, 2)
        fit = round(metrics[_classification][name][_reducer]["fit_time"].min() * 1000)
        predict = round(metrics[_classification][name][_reducer]["prediction_time"].min() * 1000)
        print("Accuracy = ", accuracy, "%", sep="")
        print("Fit time = ", fit, "ms", sep="")
        print("Predict time = ", predict, "ms", sep="")


# Main program
if __name__ == "__main__":

    # Preprocess the data
    preprocessed_data = preprocess("./data/dataset.csv", "./data/preprocessed_data")

    # Split the dataset into features and labels
    features = preprocessed_data[:, :-3]
    binary_y = preprocessed_data[:, -3]
    category_y = preprocessed_data[:, -2]
    subcategory_y = preprocessed_data[:, -1]

    # Reduce the dimension of the features using PCA
    pca_features = apply_pca(features, "./data/pca")

    # Reduce the dimension of the features using LDA
    lda_features = apply_lda(features, subcategory_y, "./data/lda")

    # Reduce the dimension of the features using UMAP
    umap3_features = apply_umap(features, 3, RANDOM_STATE, "./data/umap3")
    umap6_features = apply_umap(features, 6, RANDOM_STATE, "./data/umap6")
    umap8_features = apply_umap(features, 8, RANDOM_STATE, "./data/umap8")

    # if the metrics' dictionary already exists, then read it
    if os.path.isfile("results/metrics.pkl"):
        print("Result data already exist")
        print("Read old results ... ", end="")
        readfile = open("results/metrics.pkl", "rb")
        metrics = pickle.load(readfile)
        readfile.close()
        print("Done")

    # Prepare a dictionary for metrics if it doesn't exist
    else:
        print("Prepare results dictionary ...", end="")
        metrics = {"binary": dict(), "category": dict(), "subcategory": dict()}
        for classification in metrics:
            metrics[classification] = {"LR": dict(), "CART": dict(), "SVM": dict(),
                                       "MLP5": dict(), "MLP10": dict(), "MLP40": dict(), "MLP100": dict()}
            for algorithm in metrics[classification]:
                metrics[classification][algorithm] = {"original": dict(),
                                                      "pca3": dict(), "pca6": dict(), "pca8": dict(),
                                                      "lda3": dict(), "lda6": dict(), "lda8": dict(),
                                                      "umap3": dict(), "umap6": dict(), "umap8": dict()}
                for reducer in metrics[classification][algorithm]:
                    if classification == "binary":
                        metrics[classification][algorithm][reducer] =\
                            {"fit_time": [], "prediction_time": [], "accuracy": [],
                             "precision": [], "recall": [], "f1": []}
                    else:
                        metrics[classification][algorithm][reducer] =\
                            {"fit_time": [], "prediction_time": [], "accuracy": []}
                        metrics[classification][algorithm][reducer]["precision"] =\
                            {"micro": [], "macro": [], "weighted": []}
                        metrics[classification][algorithm][reducer]["recall"] =\
                            {"micro": [], "macro": [], "weighted": []}
                        metrics[classification][algorithm][reducer]["f1"] =\
                            {"micro": [], "macro": [], "weighted": []}
        print("Done")

    # Cross validation for the different tests
    # binary classification with no dimensionality reduction
    evaluate("binary", "original", features, binary_y)
    # category classification with no dimensionality reduction
    evaluate("category", "original", features, category_y)
    # subcategory classification with no dimensionality reduction
    evaluate("subcategory", "original", features, subcategory_y)
    # binary classification with UMAP
    evaluate("binary", "umap3", umap3_features, binary_y)
    evaluate("binary", "umap6", umap6_features, binary_y)
    evaluate("binary", "umap8", umap8_features, binary_y)
    # category classification with UMAP
    evaluate("category", "umap3", umap3_features, category_y)
    evaluate("category", "umap6", umap6_features, category_y)
    evaluate("category", "umap8", umap8_features, category_y)
    # subcategory classification with UMAP
    evaluate("subcategory", "umap3", umap3_features, subcategory_y)
    evaluate("subcategory", "umap6", umap6_features, subcategory_y)
    evaluate("subcategory", "umap8", umap8_features, subcategory_y)
    # binary classification with PCA
    evaluate("binary", "pca3", pca_features[:, :3], binary_y)
    evaluate("binary", "pca6", pca_features[:, :6], binary_y)
    evaluate("binary", "pca8", pca_features[:, :8], binary_y)
    # category classification with PCA
    evaluate("category", "pca3", pca_features[:, :3], category_y)
    evaluate("category", "pca6", pca_features[:, :6], category_y)
    evaluate("category", "pca8", pca_features[:, :8], category_y)
    # subcategory classification with PCA
    evaluate("subcategory", "pca3", pca_features[:, :3], subcategory_y)
    evaluate("subcategory", "pca6", pca_features[:, :6], subcategory_y)
    evaluate("subcategory", "pca8", pca_features[:, :8], subcategory_y)
    # binary classification with LDA
    evaluate("binary", "lda3", lda_features[:, :3], binary_y)
    evaluate("binary", "lda6", lda_features[:, :6], binary_y)
    evaluate("binary", "lda8", lda_features[:, :8], binary_y)
    # category classification with LDA
    evaluate("category", "lda3", lda_features[:, :3], category_y)
    evaluate("category", "lda6", lda_features[:, :6], category_y)
    evaluate("category", "lda8", lda_features[:, :8], category_y)
    # subcategory classification with LDA
    evaluate("subcategory", "lda3", lda_features[:, :3], subcategory_y)
    evaluate("subcategory", "lda6", lda_features[:, :6], subcategory_y)
    evaluate("subcategory", "lda8", lda_features[:, :8], subcategory_y)

    # Convert the metrics dictionary to a JSON file (the dictionary contains numpy arrays)
    print("Convert results to JSON file ...", end="")
    jsonfile = open("results/metrics.json", "w")
    json.dump(metrics, jsonfile, indent=4, cls=NumpyEncoder)
    jsonfile.close()
    print("Done")

    # Generate figures to compare the accuracy of the tested algorithms
    print("Generate figures for algorithms' accuracy comparison ...", end="")
    compared_algorithms = ["CART", "MLP40", "SVM", "LR"]
    legends = ["Decision Tree", "Neural Network (One Hidden Layer with 40 Neurons)",
               "Support Vector Machine", "Logistic Regression"]
    for reducer in REDUCERS:
        for metric in ACCURACY_METRICS:
            # Set the title
            if reducer == "original":
                title = "Accuracy Comparison for Original Data"
            elif reducer.startswith("pca") or reducer.startswith("lda"):
                title = "Accuracy Comparison for " + reducer[:3].upper() \
                        + " Reduced Data (" + reducer[3] + " Dimensions)"
            else:
                title = "Accuracy Comparison for UMAP Reduced Data (" + reducer[4] + " Dimensions)"
            compare_algorithms_accuracy(metrics, reducer, metric, compared_algorithms, legends, title)
    print("Done")

    # Generate figures to compare the accuracy of the tested neural networks
    print("Generate figures for algorithms' performance comparison ...", end="")
    for reducer in REDUCERS:
        for metric in PERFORMANCE_METRICS:
            # Set the title
            if reducer == "original":
                title = "Performance Comparison for Original Data"
            elif reducer.startswith("pca") or reducer.startswith("lda"):
                title = "Performance Comparison for " + reducer[:3].upper() \
                        + " Reduced Data (" + reducer[3] + " Dimensions)"
            else:
                title = "Performance Comparison for UMAP Reduced Data (" + reducer[4] + " Dimensions)"
            compare_algorithms_performance(metrics, reducer, metric, compared_algorithms, legends, title)
    print("Done")

    # Generate figures to compare the accuracy of the tested neural networks
    print("Generate figures for neural networks accuracy comparison ...", end="")
    compared_algorithms = ["MLP100", "MLP40", "MLP10", "MLP5"]
    legends = ["100 Neurons", "40 Neurons", "10 Neurons", "5 Neurons"]
    for reducer in REDUCERS:
        for metric in ACCURACY_METRICS:
            # Set the title
            if reducer == "original":
                title = "Accuracy of the Neural Networks for Original Data"
            elif reducer.startswith("pca") or reducer.startswith("lda"):
                title = "Accuracy of the Neural Networks for " + reducer[:3].upper() \
                        + " Reduced Data (" + reducer[3] + " Dimensions)"
            else:
                title = "Accuracy of the Neural Networks for UMAP Reduced Data (" + reducer[4] + " Dimensions)"
            compare_algorithms_accuracy(metrics, reducer, metric, compared_algorithms, legends, title)
    print("Done")

    # Generate figures to compare the accuracy of the tested neural networks
    print("Generate figures for neural networks performance comparison ...", end="")
    for reducer in REDUCERS:
        for metric in PERFORMANCE_METRICS:
            # Set the title
            if reducer == "original":
                title = "Performance of the Neural Networks for Original Data"
            elif reducer.startswith("pca") or reducer.startswith("lda"):
                title = "Performance of the Neural Networks for " + reducer[:3].upper() \
                        + " Reduced Data (" + reducer[3] + " Dimensions)"
            else:
                title = "Performance of the Neural Networks for UMAP Reduced Data (" + reducer[4] + " Dimensions)"
            compare_algorithms_performance(metrics, reducer, metric, compared_algorithms, legends, title)
    print("Done")

    # Generate figures to compare the different dimensionality reduction methods in terms of accuracy
    print("Generate figures for dimensionality reduction accuracy comparisons ...", end="")
    for algorithm, fullname, _ in MODELS:
        for classification in CLASSIFICATIONS:
            for metric in ACCURACY_METRICS:
                if metric == "accuracy" or classification == "binary":
                    compare_accuracy_dimension(metrics, classification, algorithm, fullname, metric)
                else:
                    for option in OPTIONS:
                        compare_accuracy_dimension(metrics, classification, algorithm, fullname, metric, option)
    print("Done")

    # Generate figures to compare the different dimensionality reduction methods in terms of performance
    print("Generate figures for dimensionality reduction performance comparisons ...", end="")
    for algorithm, fullname, _ in MODELS:
        for classification in CLASSIFICATIONS:
            for metric in PERFORMANCE_METRICS:
                compare_performance_dimension(metrics, classification, algorithm, fullname, metric)
    print("Done")
