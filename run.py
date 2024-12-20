import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from ANN import NeuralNetwork

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Training and Visualization CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command with visualization option
    train_parser = subparsers.add_parser('train', help='Train a new neural network')
    train_parser.add_argument('--input-size', type=int, default=2,
                            help='Number of input features (default: 2)')
    train_parser.add_argument('--hidden-layers', type=int, default=3,
                            help='Number of hidden layers (default: 3)')
    train_parser.add_argument('--output-size', type=int, default=2,
                            help='Number of output classes (default: 2)')
    train_parser.add_argument('--neurons', type=int, default=16,
                            help='Number of neurons per hidden layer (default: 16)')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                            help='Learning rate (default: 0.001)')
    train_parser.add_argument('--epochs', type=int, default=250,
                            help='Number of training epochs (default: 250)')
    train_parser.add_argument('--samples', type=int, default=10000,
                            help='Number of training samples (default: 10000)')
    train_parser.add_argument('--save-path', type=str, default='model',
                            help='Path to save the trained model (default: model)')
    train_parser.add_argument('--visualise', action='store_true',
                            help='Visualise the decision boundary after training')
    train_parser.add_argument('--resolution', type=int, default=100,
                            help='Grid resolution for visualization (default: 100)')

    
    return parser.parse_args()

def generate_data(n_samples, n_features, n_classes):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        n_redundant=0,
        n_informative=n_features,
        random_state=42
    )
    
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = one_hot_encoder.fit_transform(y.reshape(-1, 1))
    
    return X, y_one_hot, y

def visualise_predictions(model, X_train, y_train, resolution=100):
    # Only visualize if we have 2D input
    if X_train.shape[1] != 2:
        print("Visualization is only available for 2D input data")
        return
        
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    predictions = model.forward(grid_points)
    predictions = np.argmax(predictions, axis=1)
    predictions = predictions.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.4, cmap='RdYlBu')
    plt.colorbar()
    
    # Use the actual training data for scatter plot
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', alpha=0.6)
    
    plt.title('Neural Network Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    args = parse_args()
    
    if args.command == 'train':
        # Generate data
        X, y_one_hot, y = generate_data(args.samples, args.input_size, args.output_size)
        
        # Create and train network
        nn = NeuralNetwork()
        nn.createNetwork(args.input_size, args.hidden_layers, 
                        args.output_size, args.neurons, args.learning_rate)
        
        nn.train(X, y_one_hot, args.epochs)
        nn.saveWeights(args.save_path)
        
        # Visualize if requested and input is 2D
        if args.visualise:
            visualise_predictions(nn, X, y, args.resolution)

if __name__ == "__main__":
    main()