# MLZero: Machine Learning from Scratch

[![PyPI version](https://badge.fury.io/py/mlzero.svg)](https://pypi.org/project/mlzero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**MLZero** is a Python library providing a collection of machine learning algorithms implemented from scratch. The goal is to offer a clear, educational codebase for understanding the fundamentals of machine learning, with practical driver scripts and a modular design for easy extension.

## Features

- **Classifiers**: Perceptron, AdaLine, Logistic Regression, k-Nearest Neighbors (kNN), Naive Bayes, Softmax Regression
- **Clusterers**: K-Means clustering algorithm
- **Regressors**: Linear regression (closed-form and gradient descent), L1 (Lasso) and L2 (Ridge) regularization, ElasticNet, polynomial regression, multiple variable regression
- **Small Neural Nets**: Basic implementation of a multi-neuron layer
- **Decomposers**: Principal Component Analysis (PCA)
- **Metrics**: Regression and classification metrics (MAE, MSE, R², accuracy, precision, recall, F1, etc.)

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `matplotlib` for plotting purpose not necessary otherwise, recommended to have installed

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
mlzero/
├── classifiers/        # Classification algorithms and driversC/
├── clusterers/         # Clustering algorithms and driversK/
├── regressors/         # Regression algorithms and driversR/
├── decomposers/        # Dimensionality reduction and driversD/
├── metrics/            # Regression and classification metrics
├── small_neural_net/   # Multi-neuron layer implementations
└── requirements.txt    # Dependencies list
```

## Usage

Each algorithm has a corresponding driver script in its drivers subdirectory. For example:

**Run the ElasticNet regressor:**

```bash
python regressors/driversR/driverElasticNet.py
```

**Run the kNN classifier:**

```bash
python classifiers/driversC/driverKNNClassifier.py
```

## Development Status

MLZero is under active development. The codebase is modular and designed for educational purposes. Contributions for new algorithms, bug fixes, and documentation improvements are welcome.

## Contributing

Contributions are welcome! Feel free to fork the repo, submit issues, or open pull requests. Please ensure your code is well-documented and tested before submitting.

## License

This project is licensed under the MIT License.
