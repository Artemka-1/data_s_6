import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def hypothesis(X, w):
    return X @ w

def loss_function(X, y, w):
    n = len(y)
    predictions = hypothesis(X, w)
    return (1 / (2 * n)) * np.sum((predictions - y) ** 2)

def gradient_step(X, y, w, lr):
    n = len(y)
    gradient = (1 / n) * X.T @ (hypothesis(X, w) - y)
    return w - lr * gradient
def gradient_descent(X, y, lr=0.01, epochs=1000):
    w = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        w = gradient_step(X, y, w, lr)
        losses.append(loss_function(X, y, w))

    return w, losses
data = pd.read_csv("housing.csv")

X = data[["area", "bathrooms", "bedrooms"]].values
y = data["price"].values
X = np.hstack([np.ones((X.shape[0], 1)), X])
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])
w_gd, losses = gradient_descent(X, y, lr=0.01, epochs=3000)

print("Параметри (Gradient Descent):")
print(w_gd)
w_analytic = np.linalg.inv(X.T @ X) @ X.T @ y

print("Параметри (Аналітичне рішення):")
print(w_analytic)

model = LinearRegression(fit_intercept=False)
model.fit(X, y)

print("Параметри (sklearn):")
print(model.coef_)

comparison = pd.DataFrame({
    "Gradient Descent": w_gd,
    "Analytical": w_analytic,
    "Sklearn": model.coef_
})

comparison


