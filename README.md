# MLZero: Machine Learning from Scratch

## Overview

MLZero is a collection of machine learning algorithms implemented from scratch in Python. The goal of this project is to provide a deeper understanding of how these algorithms work under the hood.

## Features

- **Classifiers**: Includes implementations of Perceptron and AdaLine.
- **Clusterers**: K-Means clustering algorithm.
- **Regressors**: Linear regression (closed-form and gradient descent), L1 and L2 regularization, ElasticNet, and polynomial regression.
- **Multi-Neuron Layer**: Basic implementation of a multi-neuron layer.

## Requirements

To run this project, you need the following Python libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Directory Structure

- `classifiers/`: Contains classification algorithms and their drivers.
- `clusterers/`: Contains clustering algorithms and their drivers.
- `regressors/`: Contains regression algorithms and their drivers.
- `multi_neuron_layer/`: Contains implementations for multi-neuron layers.

## Usage

Each algorithm has a corresponding driver script in the `drivers` subdirectories. For example, to run the ElasticNet regressor:

```bash
python regressors/drivers/driverElasticNet.py
```

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License.
