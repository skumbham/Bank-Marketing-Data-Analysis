---
title: "Bank Marketing Data Analysis"
output: github_document
---

# Bank Marketing Data Analysis

This repository contains a Python analysis of the Bank Marketing dataset from the UCI Machine Learning Repository. The analysis aims to identify factors that influence a client's decision to subscribe to a term deposit, facilitating the development of more effective bank marketing strategies.

## Dataset

The dataset originates from direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls, with the objective of promoting term deposits among bank clients. The dataset provides various attributes related to clients and campaign outcomes.

For detailed information about the dataset, visit [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Analysis Overview

The Python script provided in this repository performs the following key tasks:

### Data Loading and Inspection

Loads the `bank-additional-full.csv` file and inspects its content, structure, and statistics.

### Data Visualization

Generates visualizations to understand the distributions of different client attributes within the dataset.

### Data Preprocessing

Encodes categorical variables, handles missing values (if any), splits the dataset into training and test sets, and normalizes the features.

### Model Training and Evaluation

#### Logistic Regression

Trains a logistic regression model to predict whether a client will subscribe to a term deposit. Evaluates the model's performance using metrics like accuracy, precision, recall, and F1-score. The logistic regression analysis provides valuable insights into which features are most influential in predicting the outcome.

#### Elbow Method

Utilizes the Elbow Method to determine the optimal number of clusters for K-Means clustering. This method involves plotting the sum of squared distances from each point to its assigned center for various cluster counts and identifying the 'elbow' point where the rate of decrease sharply changes. This point suggests a good balance between the number of clusters and the within-cluster sum of squares.

### Cluster Analysis

#### K-Means Clustering with Voronoi Diagram

Performs K-Means clustering to identify distinct customer segments based on their features. The analysis uses the optimal number of clusters determined by the Elbow Method. The resulting clusters are visualized using Voronoi diagrams, which partition the feature space into regions based on the nearest cluster centroids. This visualization provides a clear and intuitive way to understand the segmentation and can inform targeted marketing strategies based on cluster characteristics.

## Libraries Used

- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

## How to Run

1. Ensure you have Python installed on your system.
2. Install the required libraries: `pip install numpy pandas matplotlib seaborn scikit-learn`.
3. Clone this repository to your local machine.
4. Run the script: `python bank_marketing_analysis.py` (replace with the actual script name).

## Results

The analysis elucidates the demographic and socio-economic characteristics influencing clients' decisions on term deposit subscriptions. The logistic regression model offers insights into predictive factors, while the K-Means clustering with Voronoi diagrams provides a strategic perspective on customer segmentation, enabling more effective targeting in marketing campaigns.
