##OBJECTIVE:

To implement a Multi-Layer Perceptron (MLP) using NumPy and demonstrate its ability to solve the XOR Boolean function. XOR is a non-linearly separable problem that requires at least one hidden layer for correct classification.


##Description of the Model:

A Multi-Layer Perceptron (MLP) is a type of artificial neural network that consists of an input layer, one or more hidden layers, and an output layer. In this implementation:

  Input Layer: Two neurons representing the binary inputs.

  Hidden Layer: Four neurons acting as feature extractors, using logical AND, OR, and NOR-like operations.

  Output Layer: A single neuron that combines hidden layer outputs to compute the XOR function.

  Activation Function: A step function is used to determine neuron activation.

  Forward Propagation: Computes outputs at each layer using dot products and activation functions.

##Description of Code:

  Perceptron Function: Implements a step activation function to compute neuron outputs.

  Hidden Layer Processing: Uses four perceptrons with predefined weights and biases to extract relevant XOR features.

  Output Layer Processing: Combines hidden layer outputs with another perceptron to compute the final XOR output.

  Testing: The model is tested on the XOR truth table inputs, and results are printed.

  Performance Evaluation: Calculates accuracy based on correct classifications.

##Performance Evaluation:

  Accuracy: The model achieves 100% accuracy on the XOR problem, demonstrating that the step-based perceptron MLP can successfully classify XOR inputs.

  Loss Reduction: Explicit loss calculation is not used, but correct predictions validate the model’s effectiveness.

  Confusion Matrix (Optional): A possible future enhancement to visualize correct and incorrect classifications.

  Graphical Representation (Optional): Decision boundaries could be plotted to better understand the model’s separability.

##MY COMMENTS:

  (a) Limitations:

   The model is manually designed with fixed weights and biases, limiting its adaptability to other problems.

   The step function prevents the use of gradient-based optimization methods like backpropagation.

  (b) Scope for Improvement:

    Implement a trainable version using backpropagation to optimize weights dynamically.

    Use non-linear activation functions like sigmoid or ReLU to enable smooth learning.

    Extend the model to handle multi-class classification problems beyond XOR.

