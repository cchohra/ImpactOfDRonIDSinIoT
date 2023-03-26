# IoT Intrusion Detection System using Dimensionality Reduction Techniques
The project aims to investigate the impact of dimensionality reduction techniques on the accuracy and performance of machine learning-based intrusion detection systems in IoT environments.

# Introduction
Intrusion detection systems (IDS) are essential for ensuring the security of IoT systems. However, traditional IDS techniques are not suitable for IoT environments due to the high-dimensional nature of IoT data. In this project, we explore the impact of dimensionality reduction techniques on the accuracy and performance of machine learning-based IDS in IoT environments.

# Dependencies
The following dependencies are required to run the program:
- Python 3.x
- Scikit-learn
- Pandas
- NumPy

# Installation
Clone the repository: git clone git@github.com:cchohra/ImpactOfDRonIDSinIoT.git
**Update later.**
Install the dependencies: pip install -r requirements.txt

# Usage
Run the main.py file: python main.py
The program will load the dataset, preprocess the data, perform dimensionality reduction using various techniques, and train a machine learning model to classify the data.
The program will output the accuracy and performance metrics for each technique.

# Dataset
We used the IoTID20 dataset for this project. The dataset can be downloaded from [here](https://sites.google.com/view/iot-network-intrusion-dataset/home).

# Dimensionality Reduction Techniques
We used the following dimensionality reduction techniques in this project:
- Principal Component Analysis (PCA).
- Linear Discriminant Analysis (LDA).
- t-Distributed Stochastic Neighbor Embedding (t-SNE).

Several machine learning models were used to classify the data:
- Logistic Regression.
- Decision Tree.
- Support Vector Machine.
- Multilayer Perceptron (neural network).

# Results
**Update later.**
The results of our experiments show that t-SNE and UMAP perform better than PCA and LDA in terms of accuracy and performance.

# Conclusion
**Update later.**
In conclusion, our study shows that dimensionality reduction techniques such as t-SNE and UMAP can significantly improve the accuracy and performance of machine learning-based IDS in IoT environments.

# License
This project is licensed under the [MIT License](https://fr.wikipedia.org/wiki/Licence_MIT).

# Acknowledgments
We would like to thank the creators of the [IoTID20](https://sites.google.com/view/iot-network-intrusion-dataset/home) dataset for making it publicly available.
