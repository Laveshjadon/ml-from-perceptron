# Linear Binary Classifier: Perceptron Implementation

This repository contains a manual implementation of the Rosenblatt Perceptron algorithm (1957). The goal of this project was to move away from "black box" libraries like Scikit-Learn and understand the mathematical mechanics of convergence in linear classifiers.

## 🧠 The Mathematical Concept

The Perceptron is designed to find a separating hyperplane defined by $w \cdot x + b = 0$.

It models a single neuron:
1.  **Input:** Vector $x$
2.  **Weights:** Vector $w$ (initially random or zero)
3.  **Activation:** Step function (returns 1 if $w \cdot x + b > 0$, else -1)

### The Learning Rule (Stochastic Gradient Descent)
Unlike the analytical solution used in Linear Regression (Normal Equation), the Perceptron learns iteratively. I implemented the standard update rule:

$$w_{new} = w_{old} + \eta \cdot (y_{true} - y_{pred}) \cdot x_i$$

Where:
* $\eta$ (eta) is the learning rate.
* The term $(y_{true} - y_{pred})$ acts as the error signal.

## 🛠️ Implementation Details & Challenges

### 1. Label Encoding (The "Zero" Trap)
One specific technical challenge I encountered was data compatibility. The `make_classification` function from Scikit-Learn generates labels as `{0, 1}`.

However, the Perceptron update rule relies on the sign of the product $y \cdot f(x)$.
* If $y=0$, the term $y \cdot x$ vanishes, and weights do not update.
* **Solution:** I implemented a preprocessing step to map $0 \to -1$, ensuring that the geometry of the update vector always points in the correct direction (away from error).

### 2. Vectorized vs. Loop Implementation
For clarity, this implementation iterates through samples one by one (Stochastic approach) rather than using full batch matrix multiplication. This highlights how individual outliers affect the decision boundary during training.

## 💻 Core Algorithm Snippet

The logic resides in the custom `perceptron` function. It does not use any auto-differentiation libraries.

```python
def perceptron(X, y):
    # Weights initialization
    w1 = w2 = b = 1
    lr = 0.1
    
    # Training Loop
    for epoch in range(1000):
        for i in range(len(X)):
            
            # Linear combination
            z = w1*X[i][0] + w2*X[i][1] + b

            # Check for misclassification
            if z * y[i] < 0:
                # Update weights towards the correct vector
                w1 = w1 + lr * y[i] * X[i][0]
                w2 = w2 + lr * y[i] * X[i][1]
                b  = b  + lr * y[i]
            
    return w1, w2, b
