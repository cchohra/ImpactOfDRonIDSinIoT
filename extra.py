
# This module is used to display explained variance ratio of the dimensionality reduction techniques.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def plot_explained_variance_ratio(explained_variance, target_file):
    plt.figure(figsize=(16, 9), dpi=120)
    plt.plot(np.cumsum(explained_variance) * 100)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=22)
    plt.savefig(target_file + ".pdf", format="pdf", dpi=120)
    plt.savefig(target_file + ".jpg", format="jpg", dpi=120)


def draw_pie_chart(slices, target_file):
    plt.figure(figsize=(12, 9), dpi=120)
    plt.pie(slices, labels=["Feature 1", "Feature 2", "Feature 3"],
            autopct="%1.1f%%", startangle=140)
    plt.setp(plt.gca().texts, fontsize=14, fontweight="bold")
    plt.axis("equal")
    plt.savefig(target_file + ".pdf", format="pdf", dpi=120)
    plt.savefig(target_file + ".jpg", format="jpg", dpi=120)


if __name__ == "__main__":

    # Load the preprocessed data
    preprocessed_data = np.load("data/preprocessed_data.npy", allow_pickle=True)

    # Extract the preprocessed data
    X = preprocessed_data[:, :-3]
    subcategory_labels = preprocessed_data[:, -1]
    binary_labels = preprocessed_data[:, -3]

    # # Plot the explained variance ratio of PCA
    # print("Plot the explained variance ratio of PCA ... ", end="")
    # pca = PCA()
    # pca.fit(X)
    # pca_explained_variance_ratio = np.insert(pca.explained_variance_ratio_[0:8], 0, 0)
    # plot_explained_variance_ratio(pca_explained_variance_ratio, "figures/pca_explained_variance")
    # print("Done")
    #
    # # Plot the explained variance ratio of LDA
    # print("Plot the explained variance ratio of LDA ... ", end="")
    # lda = LinearDiscriminantAnalysis()
    # lda.fit(X, subcategory_labels)
    # lda_explained_variance_ratio = np.insert(lda.explained_variance_ratio_, 0, 0)
    # plot_explained_variance_ratio(lda_explained_variance_ratio, "figures/lda_explained_variance")
    # print("Done")

    # # Feature importance of the Decision Tree Classifier using PCA features
    # pca_features = np.load("data/pca.npy", allow_pickle=True)
    # X_train, _, y_train, _ = train_test_split(pca_features[:, 0:3], binary_labels, test_size=0.2)
    # cart = DecisionTreeClassifier(splitter="random")
    # cart.fit(X_train, y_train)
    # draw_pie_chart(cart.feature_importances_, "figures/pca_feature_importance")
    #
    # # You should use best splitter foe the next code to be relevant (slower than paper results)
    # split_features = cart.tree_.feature
    # split_features = split_features[split_features != -2]
    # _, counts = np.unique(split_features, return_counts=True)
    # print("Feature 1: ", counts[0], "Feature 2: ", counts[1], "Feature 3: ", counts[2])
    #
    # # Feature importance of the Decision Tree Classifier using LDA features
    # lda_features = np.load("data/lda.npy", allow_pickle=True)
    # X_train, _, y_train, _ = train_test_split(lda_features[:, 0:3], binary_labels, test_size=0.2)
    # cart = DecisionTreeClassifier(splitter="random")
    # cart.fit(X_train, y_train)
    # draw_pie_chart(cart.feature_importances_, "figures/lda_feature_importance")
    #
    # # You should use best splitter foe the next code to be relevant (slower than paper results)
    # split_features = cart.tree_.feature
    # split_features = split_features[split_features != -2]
    # _, counts = np.unique(split_features, return_counts=True)
    # print("Feature 1: ", counts[0], "Feature 2: ", counts[1], "Feature 3: ", counts[2])
    #
    # # Feature importance of the Decision Tree Classifier using UMAP features
    # umap_features = np.load("data/umap3.npy", allow_pickle=True)
    # X_train, _, y_train, _ = train_test_split(umap_features, binary_labels, test_size=0.2)
    # cart = DecisionTreeClassifier(splitter="random")
    # cart.fit(X_train, y_train)
    # draw_pie_chart(cart.feature_importances_, "figures/umap_feature_importance")
    #
    # # You should use best splitter foe the next code to be relevant (slower than paper results)
    # split_features = cart.tree_.feature
    # split_features = split_features[split_features != -2]
    # _, counts = np.unique(split_features, return_counts=True)
    # print("Feature 1: ", counts[0], "Feature 2: ", counts[1], "Feature 3: ", counts[2])

    # Generate confusion matrix for subcategory classification with decision tree using original features
    X_train, X_test, y_train, y_test = train_test_split(X, subcategory_labels, test_size=0.2)
    cart = DecisionTreeClassifier(splitter="random")
    cart.fit(X_train, y_train)
    y_pred = cart.predict(X_test)
    mat = confusion_matrix(y_test, y_pred)
    print(mat)
    print("Accuracy of the classifier: ", np.trace(mat) / np.sum(mat) * 100)
    mat = mat / mat.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(16, 9), dpi=120)
    plt.matshow(mat, cmap="Blues")
    plt.colorbar()
    plt.savefig("figures/confusion_matrix_original.pdf", format="pdf", dpi=120)
    plt.savefig("figures/confusion_matrix_original.jpg", format="jpg", dpi=120)

    # Generate confusion matrix for subcategory classification with decision tree using UMAP 3 dimensions data
    umap_features = np.load("data/umap3.npy", allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(umap_features, subcategory_labels, test_size=0.2)
    cart = DecisionTreeClassifier(splitter="random")
    cart.fit(X_train, y_train)
    y_pred = cart.predict(X_test)
    mat = confusion_matrix(y_test, y_pred)
    print(mat)
    print("Accuracy of the classifier: ", np.trace(mat) / np.sum(mat) * 100)
    mat = mat / mat.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(16, 9), dpi=120)
    plt.matshow(mat, cmap="Blues")
    plt.colorbar()
    plt.savefig("figures/confusion_matrix_umap.pdf", format="pdf", dpi=120)
    plt.savefig("figures/confusion_matrix_umap.jpg", format="jpg", dpi=120)

