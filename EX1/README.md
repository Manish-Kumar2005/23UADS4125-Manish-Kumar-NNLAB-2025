OBJECTIVE:
To implement the Perceptron Learning Algorithm using NumPy in Python and evaluate the performance of a single-layer perceptron for the NAND and XOR truth tables as input datasets.

Description of the Model:

A Perceptron is a fundamental building block of artificial neural networks, primarily used for binary classification tasks. It learns to classify input patterns by adjusting weights based on the error observed during training. The model consists of:

Input Layer: Two neurons representing binary input values.

Single-Layer Perceptron: Implements the learning rule with adjustable weights.

Activation Function: Uses a step function (threshold function) to determine the output.

Learning Process: Updates weights using the perceptron learning rule iteratively.

For this task, we apply the perceptron model to:

NAND Logic Gate: A linearly separable function that a single perceptron can learn.

XOR Logic Gate: A non-linearly separable function that a single-layer perceptron fails to classify correctly.

Description of Code:

Step 1: Required Libraries Import

Imports essential libraries such as NumPy for computations and Matplotlib for visualization.

Uses accuracy_score and confusion_matrix from sklearn.metrics to evaluate the model.

Step 2: Perceptron Class Implementation

Defines a perceptron class with weight initialization, activation function, and learning algorithm.

The perceptron updates weights iteratively using the perceptron learning rule.

Step 3: NAND & XOR Truth Table

Defines datasets for NAND and XOR operations.

NAND is linearly separable, while XOR requires a multi-layer approach.

Step 4: Training & Prediction

Trains the perceptron on the NAND dataset and makes predictions.

Trains the perceptron on the XOR dataset and makes predictions.

Step 5: Performance Evaluation

Computes accuracy for both NAND and XOR gates.

Displays confusion matrices for visualizing classification performance.

Performance Evaluation:

NAND Gate: Since NAND is linearly separable, the perceptron successfully classifies all inputs with high accuracy.

XOR Gate: The perceptron fails to classify XOR correctly as it is a non-linearly separable function.

Accuracy Metrics: Measured using accuracy_score() to determine classification effectiveness.

Confusion Matrices: Displayed using ConfusionMatrixDisplay() to analyze misclassifications.


MY COMMENTS:

NAND Function: The perceptron successfully classifies NAND outputs as it is linearly separable.
XOR Function: The perceptron fails since XOR is not linearly separable.

The step activation function prevents smooth learning, limiting generalization.
