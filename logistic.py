import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

X = df[["X_1", "X_2"]].values
y = df["y"].values.reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y_hat, y):
    return np.mean(-y * np.log2(y_hat) - (1 - y) * np.log2(1 - y_hat))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def LogisticRegression(X, y, learning_rate=0.01, iterations=10000):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0

    for i in range(iterations):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        d_Z = y_hat - y
        d_w = np.dot(X.T, d_Z) / m
        d_b = np.sum(d_Z) / m

        w = w - learning_rate * d_w
        b = b - learning_rate * d_b

        if i % 100 == 0:
            l = loss(y_hat, y)
            print(f"Iteration {i}: Loss = {l:.4f}")

    return w, b

def predict(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

w, b = LogisticRegression(X, y)
y_pred = predict(X, w, b)

acc = accuracy(y, y_pred)
print(f"Accuracy: {acc:.4f}")


