# PlayTennis Classification using Naive Bayes

Predict whether a person will play tennis based on weather conditions using a Naive Bayes classifier with Laplace smoothing.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview
This Python script uses `pandas` and `numpy` to:
- Load a dataset (`dataset.csv`).
- Compute prior and Laplace-smoothed likelihood probabilities.
- Classify instances and log results to `classification_log.txt`.
- Calculate accuracy and confusion matrix.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/MuhammedNael/naive-bayes-classifier
    cd naive-bayes-classifier

2. **Install dependencies**:
    Python 3.x
    pandas
    numpy

3. **Add your dataset**:
    Place dataset.csv in the project directory with these columns:
    Outlook, Temperature, Humidity, Wind, PlayTennis.

## Usage

1. **Run the classifier**:
    python naive_bayes_classifier.py

2. **Output**:
    * Results printed to the console.
    * Detailed logs saved to classification_log.txt.

## Results
    * Example output
    Accuracy: 0.85
    Confusion Matrix:
        True Positives: 5
        False Positives: 1
        True Negatives: 3
        False Negatives: 1