Quantum and Classical Machine Learning on Custom Dataset
This project demonstrates the implementation of both classical and quantum machine learning models on a custom dataset. The dataset contains bounding box coordinates and class labels, which are used to train and evaluate Support Vector Classifier (SVC) and Variational Quantum Classifier (VQC) models. The project also includes visualization of the results.

Table of Contents
Introduction
Requirements
Installation
Usage
Dataset
Model Training and Evaluation
Visualization
Acknowledgements
Introduction
The goal of this project is to compare the performance of classical and quantum machine learning models on a custom dataset. The dataset includes bounding box coordinates and class labels, which are used to train both SVC and VQC models. The project provides a comprehensive approach to preprocess the dataset, train the models, evaluate their performance, and visualize the results.

Requirements
Python 3.7+
Libraries:
pandas
numpy
scikit-learn
matplotlib
seaborn
pennylane
pennylane-qiskit (optional, if using Qiskit as a backend for PennyLane)


frame	xmin	xmax	ymin	ymax	class_id
1478019952686311006.jpg	237	251	143	155	1
1478019952686311006.jpg	437	454	120	186	3
...	...	...	...	...	...
xmin, xmax, ymin, ymax are the bounding box coordinates.
class_id is the class label for the object in the bounding box.
Model Training and Evaluation
The script performs the following steps:

Preprocessing: Loads and preprocesses the dataset, including standardization of features.
Classical Model (SVC):
Trains a Support Vector Classifier on the dataset.
Evaluates the model using accuracy, classification report, and confusion matrix.
Visualizes the confusion matrix.
Quantum Model (PennyLane):
Defines a quantum circuit and trains a Variational Quantum Classifier using PennyLane.
Evaluates the model using accuracy, classification report, and confusion matrix.
Visualizes the confusion matrix.
Visualization:
Reduces the dataset to 2 dimensions using PCA for visualization purposes.
Plots decision boundaries of the quantum classifier.
Visualization
The script includes visualization of:

Confusion matrices for both SVC and VQC models.
Decision boundaries of the quantum classifier on PCA-reduced data.
Acknowledgements
This project utilizes the following libraries:

Pandas
NumPy
scikit-learn
Matplotlib
Seaborn
PennyLane
