# 🧠 ML Models From Scratch: The Perceptron

Hi there! 👋 Welcome to my repository.

This project is my journey into understanding the **mathematics behind Machine Learning**. Instead of just importing `LinearRegression` from libraries and calling `.fit()`, I wanted to build the foundational algorithms from scratch to see what's actually happening "under the hood."

## 🚀 What is this project?
This is a raw, manual implementation of the **Rosenblatt Perceptron Algorithm** in Python.

The goal was simple:
1.  Generate synthetic data (two distinct groups of points).
2.  Write a linear classifier **without** using `sklearn`'s built-in models.
3.  Visually prove that my math works by drawing the Decision Boundary (the separating line).

## 🛠️ How I built it
I used **Python** with `numpy` for the math and `matplotlib` for the visuals.

### 1. The Data
I used `make_classification` to generate a dataset with 2 features.
* **Challenge:** The data originally came with `0` and `1` labels.
* **The Fix:** Since the Perceptron math requires `1` and `-1` to calculate updates correctly, I wrote a small preprocessing step to convert the labels.

### 2. The Algorithm (The "Perceptron Trick")
I implemented the classic update rule using **Stochastic Gradient Descent (SGD)**.
* The model looks at one point at a time.
* If it classifies the point correctly? Do nothing.
* If it makes a mistake? Shift the line slightly towards the point.

```python
# The core logic I implemented
if prediction * actual < 0:
    weights = weights + learning_rate * actual * input_features
