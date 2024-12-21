# Neural Network Implementation

A flexible implementation of a feedforward neural network with customizable architecture, written in Python. This implementation includes a command-line interface for training and visualizing the network's decision boundaries on classification tasks.

## Features

- Customizable neural network architecture with variable:
  - Input size
  - Number of hidden layers
  - Neurons per hidden layer
  - Output size
- ReLU activation for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Built-in visualization of decision boundaries for 2D inputs
- Model saving and loading capabilities

## Requirements

```
numpy
scikit-learn
matplotlib
pickle
argparse
```

## Usage

The program can be run from the command line using various commands and options.

### Training a New Network

Basic usage:
```bash
python run.py train
```

With custom parameters:
```bash
python run.py train --input-size 2 --hidden-layers 3 --output-size 2 --neurons 16 --learning-rate 0.001 --epochs 250 --samples 10000 --visualise
```

Example of visualisation:
<img width="945" alt="Screenshot 2024-12-21 at 12 26 03 am" src="https://github.com/user-attachments/assets/94ff75c4-440a-4b60-8293-3415aec69324" />


### Command Line Arguments

- `--input-size`: Number of input features (default: 2)
- `--hidden-layers`: Number of hidden layers (default: 3)
- `--output-size`: Number of output classes (default: 2)
- `--neurons`: Number of neurons per hidden layer (default: 16)
- `--learning-rate`: Learning rate (default: 0.001)
- `--epochs`: Number of training epochs (default: 250)
- `--samples`: Number of training samples (default: 10000)
- `--save-path`: Path to save the trained model (default: 'model')
- `--visualise`: Flag to visualize decision boundary after training
- `--resolution`: Grid resolution for visualization (default: 100)


## Example Output

When training with visualization enabled (for 2D inputs), the program will:
1. Train the network while displaying epoch-wise loss and accuracy
2. Save the trained model to the specified path
3. Display a plot showing the decision boundaries and training data points

## Model Persistence

Models can be saved and loaded using pickle serialization:
- Saving: Automatically done after training using the specified `--save-path`
- Loading: Use the `loadModel()` method from the `NeuralNetwork` class

## Limitations

- Visualization is only available for 2D input data
- Currently supports classification tasks only
- Uses fixed ReLU activation for hidden layers
- Uses fixed softmax activation for output layer
