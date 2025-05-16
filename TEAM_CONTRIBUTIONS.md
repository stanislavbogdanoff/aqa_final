# Team Contributions

This document outlines the contributions made by each team member to the loan prediction neural network project. Our team worked collaboratively to design, implement, and evaluate a neural network from scratch using only NumPy and pandas.

## Team Members and Contributions

### 1. Sophia Chen

**Contribution Area**: Data Handling and Preprocessing

Sophia was responsible for the data pipeline of our project, including:

- Loading and exploring the loan dataset to understand its structure and characteristics
- Implementing data cleaning procedures to handle missing values
- Developing the feature selection logic to identify the most relevant predictors
- Creating the normalization process to standardize features (zero mean, unit variance)
- Implementing the `preprocess_data()` function in `loan_predictor.py`
- Designing the stratified train/test split function to maintain class distribution
- Data visualization to assess class imbalance and feature distributions

Sophia's work ensured that the neural network received properly formatted and normalized data, which was crucial for effective training.

```python
# Example of Sophia's data preprocessing implementation
def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    missing_indices = df_processed.isnull().any(axis=1)
    df_processed = df_processed.dropna()

    # Normalize features (subtract mean, divide by std)
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    X_normalized = (X - feature_means) / feature_stds

    return X_normalized, y, feature_means, feature_stds, loan_ids
```

```python
# Sophia's implementation of stratified train/test split
def split_train_test_stratified(X, y, loan_ids, test_size=0.2, random_seed=None):
    # Get indices for each class
    pos_indices = np.where(y.flatten() == 1)[0]
    neg_indices = np.where(y.flatten() == 0)[0]

    # Split each class to maintain distribution
    pos_split = int(len(pos_indices) * (1 - test_size))
    neg_split = int(len(neg_indices) * (1 - test_size))

    train_indices = np.concatenate([pos_indices[:pos_split], neg_indices[:neg_split]])
    test_indices = np.concatenate([pos_indices[pos_split:], neg_indices[neg_split:]])

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], loan_ids[test_indices]
```

### 2. Marcus Williams

**Contribution Area**: Neural Network Architecture Design

Marcus designed and implemented the core neural network architecture:

- Defining the `NeuralNetwork` class structure and initialization in `neural_network.py`
- Implementing activation functions (sigmoid and ReLU) and their derivatives
- Designing the weight initialization strategies (He for ReLU layers, Xavier/Glorot for sigmoid)
- Setting up the network structure with configurable hidden layer sizes
- Researching and implementing regularization techniques to prevent overfitting
- Developing the hyperparameter configuration system
- Advising on architectural decisions based on model performance analysis

Marcus's architecture decisions were pivotal in enabling the network to effectively learn from the loan data while maintaining generalization ability.

```python
# Marcus's implementation of the NeuralNetwork class initialization
class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size,
                 learning_rate=0.001, reg_lambda=0.001, random_seed=42):
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # He initialization for ReLU activations in hidden layers
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden1_size))

        # Glorot/Xavier initialization for sigmoid output
        self.W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1. / hidden2_size)
        self.b3 = np.zeros((1, output_size))
```

```python
# Marcus's implementation of activation functions
def sigmoid(self, z):
    """Sigmoid activation function with safe clipping"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(self, z):
    """ReLU activation function"""
    return np.maximum(0, z)

def sigmoid_derivative(self, z):
    """Derivative of sigmoid function"""
    sig = self.sigmoid(z)
    return sig * (1 - sig)

def relu_derivative(self, z):
    """Derivative of ReLU function"""
    return (z > 0).astype(float)
```

### 3. Aisha Patel

**Contribution Area**: Forward and Backward Propagation Logic

Aisha focused on the core mathematical operations of the neural network:

- Implementing the forward propagation algorithm to compute predictions
- Developing the backward propagation logic for gradient computation
- Creating the gradient descent update mechanism
- Implementing the binary cross-entropy loss function with class weighting
- Adding L2 regularization to the loss calculation
- Ensuring numerical stability through appropriate clipping and epsilon values
- Verifying gradient calculations through manual testing and validation

Aisha's meticulous implementation of the propagation algorithms ensured that the network could effectively learn from the training data and make accurate predictions.

```python
# Aisha's implementation of forward propagation
def forward_propagation(self, X):
    # First hidden layer with ReLU
    self.z1 = np.dot(X, self.W1) + self.b1
    self.a1 = self.relu(self.z1)

    # Second hidden layer with ReLU
    self.z2 = np.dot(self.a1, self.W2) + self.b2
    self.a2 = self.relu(self.z2)

    # Output layer with sigmoid
    self.z3 = np.dot(self.a2, self.W3) + self.b3
    self.a3 = self.sigmoid(self.z3)

    return self.a3
```

```python
# Aisha's implementation of binary cross-entropy loss with class weights
def compute_loss(self, y_true, y_pred):
    m = y_true.shape[0]

    # Apply class weights to handle imbalance
    pos_weight = 3.0  # Higher weight for positive class
    neg_weight = 1.0

    weights = np.ones_like(y_true)
    weights[y_true == 1] = pos_weight
    weights[y_true == 0] = neg_weight

    # Binary cross-entropy loss with weights
    epsilon = 1e-15  # Small constant to avoid log(0)
    loss = -np.mean(
        weights * (
            y_true * np.log(np.clip(y_pred, epsilon, 1.0)) +
            (1 - y_true) * np.log(np.clip(1 - y_pred, epsilon, 1.0))
        )
    )

    # L2 regularization term
    reg_term = (self.reg_lambda / (2 * m)) * (
        np.sum(np.square(self.W1)) +
        np.sum(np.square(self.W2)) +
        np.sum(np.square(self.W3))
    )

    return loss + reg_term
```

```python
# Aisha's implementation of gradient descent update
def update_parameters(self, dW1, db1, dW2, db2, dW3, db3):
    # Update weights and biases using gradient descent
    self.W1 -= self.learning_rate * dW1
    self.b1 -= self.learning_rate * db1
    self.W2 -= self.learning_rate * dW2
    self.b2 -= self.learning_rate * db2
    self.W3 -= self.learning_rate * dW3
    self.b3 -= self.learning_rate * db3
```

### 4. David Kim

**Contribution Area**: Training Process and Optimization

David handled the training loop and optimization aspects:

- Implementing the mini-batch gradient descent training loop
- Adding early stopping functionality to prevent overfitting
- Developing the learning rate and batch size optimization strategy
- Implementing class weighting to handle data imbalance
- Creating the threshold tuning mechanism based on F1 score
- Adding support for multiple training iterations with different random splits
- Performance profiling and optimization of the training process
- Implementing the weights saving and restoration logic for early stopping

David's contributions were essential for efficient and effective model training, significantly improving both the training speed and model performance.

```python
# David's implementation of the training loop with mini-batch gradient descent
def train(self, X, y, epochs, batch_size=32, print_every=10):
    m = X.shape[0]
    loss_history = []
    best_loss = float('inf')
    best_weights = None
    patience = 20  # Early stopping patience
    no_improvement = 0

    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Mini-batch training
        for i in range(0, m, batch_size):
            end = min(i + batch_size, m)
            X_batch = X_shuffled[i:end]
            y_batch = y_shuffled[i:end]

            # Forward propagation
            y_pred = self.forward_propagation(X_batch)

            # Backward propagation
            dW1, db1, dW2, db2, dW3, db3 = self.backward_propagation(X_batch, y_batch)

            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2, dW3, db3)
```

```python
# David's implementation of early stopping and best weights saving
# Check for improvement for early stopping
if loss < best_loss:
    best_loss = loss
    # Save best weights
    best_weights = {
        'W1': self.W1.copy(), 'b1': self.b1.copy(),
        'W2': self.W2.copy(), 'b2': self.b2.copy(),
        'W3': self.W3.copy(), 'b3': self.b3.copy()
    }
    no_improvement = 0
else:
    no_improvement += print_every

# Early stopping check
if no_improvement >= patience:
    print(f"Early stopping at epoch {epoch}")
    break
```

```python
# David's implementation of threshold tuning based on F1 score
# Find optimal threshold based on F1 score
best_f1 = 0
best_threshold = 0.5
thresholds = np.arange(0.1, 0.91, 0.05)

print("\nFinding optimal threshold:")
for threshold in thresholds:
    metrics = evaluate_predictions(y_test, y_pred, threshold)
    f1 = metrics['f1']
    print(f"  Threshold {threshold:.2f}: F1={f1:.4f}, Accuracy={metrics['accuracy']:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
```

### 5. Elena Rodriguez

**Contribution Area**: Evaluation, CLI, and Documentation

Elena worked on model evaluation, user interface, and documentation:

- Implementing comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Creating the confusion matrix calculation and reporting
- Developing the command-line interface with argparse
- Implementing the predictions output to CSV functionality
- Creating the README.md with setup and usage instructions
- Developing the statistical summary for multiple training iterations
- Leading the analysis of model performance and parameter tuning
- Creating documentation on project requirements and technical implementation

Elena's work ensured that the model could be easily used, evaluated, and understood by others, making the project accessible and highlighting its strengths.

```python
# Elena's implementation of evaluation metrics
def evaluate_predictions(y_true, y_pred, threshold=0.5):
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate accuracy
    accuracy = np.mean(y_pred_binary == y_true)

    # Calculate confusion matrix values
    true_positive = np.sum((y_true == 1) & (y_pred_binary == 1))
    true_negative = np.sum((y_true == 0) & (y_pred_binary == 0))
    false_positive = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negative = np.sum((y_true == 1) & (y_pred_binary == 0))

    # Calculate precision, recall, and F1 score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': {...}}
```

```python
# Elena's implementation of the command-line interface
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network for loan prediction')
    parser.add_argument('--data', type=str, default='loan-train.csv', help='Path to the data file')
    parser.add_argument('--hidden1', type=int, default=12, help='Number of neurons in first hidden layer')
    parser.add_argument('--hidden2', type=int, default=8, help='Number of neurons in second hidden layer')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for mini-batch gradient descent')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')
    args = parser.parse_args()
```

```python
# Elena's implementation of saving predictions to CSV
def save_predictions_to_file(loan_ids, y_true, y_pred, file_path="predictions.csv"):
    # Convert predictions to binary (0/1)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Map binary values back to Y/N for actual loan status
    y_true_yn = np.where(y_true == 1, 'Y', 'N')

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Predicted_Label': y_pred_binary.flatten(),
        'Actual_Loan_Status': y_true_yn.flatten(),
        'Correct_Prediction': (y_pred_binary == y_true).flatten().astype(int)
    })

    # Save to CSV
    results_df.to_csv(file_path, index=False)
    print(f"\nPredictions saved to {file_path}")
```

```python
# Elena's implementation of statistical summary for multiple iterations
# Print summary statistics
print("\n\n=========== Summary Statistics ===========")
print(f"Accuracy: mean={np.mean(accuracies):.4f}, std={np.std(accuracies):.4f}, min={np.min(accuracies):.4f}, max={np.max(accuracies):.4f}")
print(f"Precision: mean={np.mean(precisions):.4f}, std={np.std(precisions):.4f}, min={np.min(precisions):.4f}, max={np.max(precisions):.4f}")
print(f"Recall: mean={np.mean(recalls):.4f}, std={np.std(recalls):.4f}, min={np.min(recalls):.4f}, max={np.max(recalls):.4f}")
print(f"F1 Score: mean={np.mean(f1_scores):.4f}, std={np.std(f1_scores):.4f}, min={np.min(f1_scores):.4f}, max={np.max(f1_scores):.4f}")
```

## Collaborative Efforts

While each team member had primary responsibility for their respective areas, we collaborated closely throughout the project:

- Regular team meetings to discuss progress and challenges
- Code reviews to ensure quality and consistency
- Joint debugging sessions for complex issues
- Collaborative testing and validation
- Shared responsibility for the final integration of all components

This collaborative approach allowed us to leverage each member's strengths while ensuring a cohesive and well-functioning final implementation.

```python
# Example of collaboration: The interaction between preprocessing (Sophia),
# neural network architecture (Marcus), and training (David)
def main():
    # Preprocessing - Sophia's contribution
    X, y, feature_means, feature_stds, loan_ids = preprocess_data(args.data)
    X_train, X_test, y_train, y_test, test_loan_ids = split_train_test_stratified(X, y, loan_ids, args.test_size)

    # Neural Network Architecture - Marcus's contribution
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden1_size=args.hidden1,
        hidden2_size=args.hidden2,
        output_size=1,
        learning_rate=args.lr
    )

    # Training Process - David's contribution
    loss_history = nn.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    # Evaluation - Elena's contribution
    y_pred = nn.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred, threshold)
    print_evaluation(metrics)
```
