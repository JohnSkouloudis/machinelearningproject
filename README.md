# Python scripts

This project contains several scripts and resources developed 
for a machine learning (and signal processing) course project. 
The repository includes Python scripts, a Jupyter Notebook, a dataset, and supporting files. Each file is briefly described below.

## File Description 

### Machine Learning Project:
  The project demonstrates exploratory data analysis and builds a logistic regression model
  to predict whether users make a purchase (indicated by the "Revenue" column).
  
  - `project2_dataset.csv`  
    The primary dataset used throughout the project. It contains various features
    (e.g., Month, Browser, OperatingSystems, Region, TrafficType, VisitorType, and Revenue)
    that are used for both exploratory data analysis and predictive modeling, where "Revenue" serves as the target variable.
  - `eda.py`    
    The exploratory data analysis script reads the dataset and computes key summary statistics.
    It counts the total number of records, calculates how many users made a purchase (based on the "Revenue" column),
    determines the purchase percentage, and computes the accuracy of a baseline model that always predicts no purchase.
  - `linear_classification.py`    
    This script implements a linear classification pipeline using logistic regression.
    It preprocesses the data by dropping and encoding selected features, scales the remaining features using MinMaxScaler,
    splits the data into training and test sets, trains the model, and finally evaluates its performance by printing training/testing accuracy and displaying a confusion matrix.

### Signal Processing
  - `pam.py`  
    This file demonstrates a basic Pulse Amplitude Modulation (PAM) analysis.
    It includes functions to compute the inverse Q-function using the complementary error function and calculates the Signal-to-Noise Ratio
    in dB (SNRb) for a given bit error probability and modulation order. The script then generates a plot showing how SNRb varies with different modulation orders.
  - `ppm.py`    
    This script implements Pulse Position Modulation (PPM). It converts an input string into its corresponding ASCII bit sequence,
    segments the bits into symbols based on a chosen modulation order, and maps these symbols to Gray-coded PPM values. Finally, it
    generates and displays waveform plots for various modulation orders, illustrating how the bit sequence is represented in a PPM format.
    
### it2021091_pretrained_models.ipynb
This Jupyter Notebook serves as an experimental platform for working with pretrained models.  
The notebook includes code cells with model setup, predictions, and visualizations that help compare different model outcomes, making it 
a valuable resource for understanding transfer learning concepts within the AI course project.

### Logistic Regression implementation:
  The LogisticRegression implementation folder contains a collection of scripts and a document that together support the exploration and evaluation of logistic regression models. 
  The folder is organized into several experiments, a dataset generator, the core logistic regression implementation, and the accompanying assignment document. 
  Below is a brief description of each file:
  
  - `generate_dataset.py`  
    A utility script designed to create or preprocess the dataset used in the experiments.
    This file contains functions to help create and visualize experiments on the Logistic Regression. 
  - `logistic_regression.py`    
    Contains the implementation class of the logistic regression algorithm.
    It includes data preprocessing, model training, prediction, and evaluation routines.
    This file serves as the backbone for the experiments conducted in the other scripts.
  - `experiment1.py`    
    Implements the first experiment using logistic regression. It uses the `generate_binary_problem` function
    to test the logistic regression implementation for different linear separable problems.
  - `experiment2.py`    
    This script uses the breast cancer dataset from [scikit-learn](https://scikit-learn.org/stable/) to test how well
    the logistic regression implementation performs on a real dataset,by calculating the mean of the accuracy and the standard deviation for 20
    iterations. 
  - `experiment3.py`    
    This script performs the same test as `experiment2` using the Logistic Regression class from [scikit-learn](https://scikit-learn.org/stable/) instead.
    The purpose of this experiment is to compare the performance of the two implementations of the algorithm.

## Setup
```
pip install -r requirements.txt
```

##  Create a requirements.txt file
```
pip freeze > requirements.txt
```
