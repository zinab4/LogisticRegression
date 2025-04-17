import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=1000, optimizer_type='batch', batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15  # To avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iterations):
            if self.optimizer_type == 'batch':
                linear_model = np.dot(X, self.weights) + self.bias
                y_pred = self.sigmoid(linear_model)
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)

            elif self.optimizer_type == 'stochastic':
                for i in range(n_samples):
                    xi = X[i, :].reshape(1, -1)
                    yi = y[i]
                    linear_model = np.dot(xi, self.weights) + self.bias
                    y_pred = self.sigmoid(linear_model)
                    dw = np.dot(xi.T, (y_pred - yi))
                    db = y_pred - yi
                    self.weights -= self.learning_rate * dw.flatten()
                    self.bias -= self.learning_rate * db
                y_pred = self.predict(X)
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)

            elif self.optimizer_type == 'mini-batch':
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]
                    linear_model = np.dot(X_batch, self.weights) + self.bias
                    y_pred = self.sigmoid(linear_model)
                    dw = (1 / len(X_batch)) * np.dot(X_batch.T, (y_pred - y_batch))
                    db = (1 / len(X_batch)) * np.sum(y_pred - y_batch)
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                y_pred = self.predict(X)
                loss = self.compute_loss(y, y_pred)
                self.losses.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict_class(self, X):
        probas = self.predict(X)
        return np.where(probas >= 0.5, 1, 0)


X, y = make_classification(n_samples=600, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=66)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"==========================")

# Try different optimizers
optimizers = ['batch', 'stochastic', 'mini-batch']
results = {}

for opt in optimizers:
    model = LogisticRegressionCustom(learning_rate=0.01, n_iterations=1000, optimizer_type=opt)
    model.fit(X_train, y_train)
    y_pred = model.predict_class(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    results[opt] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Report": report,
        "Losses": model.losses,
        "Confusion Matrix": cm
    }

print(f"==========================")
for opt in optimizers:
    print(f"\nOptimizer: {opt.upper()}")
    print(f"Accuracy: {results[opt]['Accuracy'] * 100:.2f}%")
    print(f"Precision: {results[opt]['Precision'] * 100:.2f}%")
    print(f"Recall: {results[opt]['Recall'] * 100:.2f}%")
    print(f"F1 Score: {results[opt]['F1 Score'] * 100:.2f}%")
    print(f"==========================")
    print("\nClassification Report:")
    print(results[opt]['Report'])

    # Confusion Matrix
    ConfusionMatrixDisplay(results[opt]['Confusion Matrix']).plot()
    plt.title(f"Confusion Matrix - {opt.title()} GD")
    plt.show()

    # Loss Curve
    plt.plot(results[opt]['Losses'])
    plt.title(f"Loss Curve - {opt.title()} GD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Decision Visualization
    y_pred = model.predict_class(X_test)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', edgecolor='k')
    plt.title(f"Prediction using {opt.title()} Gradient Descent")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    
#Comparison  

plt.figure(figsize=(10, 6))
for opt in optimizers:
    plt.plot(results[opt]['Losses'], label=f"{opt.title()} GD")
plt.title("Loss vs Epochs for Different Optimizers")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Extract the values for each optimizer
batch_vals      = [results['batch'][m]      for m in metrics]
stochastic_vals = [results['stochastic'][m] for m in metrics]
mini_vals       = [results['mini-batch'][m] for m in metrics]

# Compute x locations for the groups
x = np.arange(len(metrics))   # [0, 1, 2, 3]
width = 0.25                  # Width of each bar

plt.figure(figsize=(10, 6))
plt.bar(x - width, batch_vals,      width, label='Batch GD')
plt.bar(x,         stochastic_vals, width, label='Stochastic GD')
plt.bar(x + width, mini_vals,       width, label='Mini-batch GD')

# Labeling
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Performance Metrics Comparison')
plt.legend(title="Optimizer")
plt.grid(axis='y')
plt.show()


