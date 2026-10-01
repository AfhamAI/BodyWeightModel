# Body Weight Prediction Model

A simple machine learning project that predicts a person's **body mass (kg)** using their **height (cm)** and **age (years)**.

This was my first attempt at building a machine learning model **from scratch using NumPy**, without relying on Scikit-Learn or other machine-learning libraries.

The goal was to understand what happens behind the scenes in a basic linear regression model — from preparing the data to standardizing features, calculating the error, computing gradients, and updating the model parameters.

## Project Overview

The model uses two input features:

* **Height**
* **Age**

And predicts:

* **Mass / Weight**

The model follows this basic pipeline:

```text
Height + Age
      ↓
Feature Matrix X
      ↓
Standardization
      ↓
Linear Regression
      ↓
Prediction
      ↓
Error
      ↓
MSE
      ↓
Gradient Descent
      ↓
Updated Weights & Bias
      ↓
Final Prediction
```

## Model

The model is a manually implemented linear regression model:

```text
ŷ = Xw + b
```

For two features:

```text
ŷ = w₁x₁ + w₂x₂ + b
```

Where:

* `x₁` = height
* `x₂` = age
* `w₁`, `w₂` = learned weights
* `b` = bias
* `ŷ` = predicted mass

## Dataset

The dataset contains:

* **60 observations**
* **2 input features**
* **1 target variable**

| Feature | Description     | Unit  |
| ------- | --------------- | ----- |
| Height  | Person's height | cm    |
| Age     | Person's age    | years |
| Mass    | Target value    | kg    |

The data is represented as:

```python
X = np.column_stack((height, age))
y = mass
```

Therefore:

```text
X.shape = (60, 2)
y.shape = (60,)
```

## Feature Standardization

Before training, the input features are standardized:

```python
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_scaled = (X - X_mean) / X_std
```

This transforms the features onto comparable scales.

Conceptually:

```text
x_scaled = (x - mean) / std
```

This is important because height and age have different numerical ranges.

## Loss Function

The model uses **Mean Squared Error (MSE)** to measure prediction error:

```text
MSE = 1/n Σ(ŷ - y)²
```

In NumPy:

```python
mse = np.mean(error ** 2)
```

where:

```python
error = y_pred - y
```

## Gradient Descent

The model learns its parameters using gradient descent.

The gradients are calculated as:

```python
dw = (2 / len(X_scaled)) * (X_scaled.T @ error)

db = (2 / len(X_scaled)) * np.sum(error)
```

The parameters are then updated:

```python
w = w - learning_rate * dw

b = b - learning_rate * db
```

Training configuration:

```python
learning_rate = 0.01
epochs = 5000
```

The process repeats for each epoch, gradually adjusting `w` and `b` to reduce the MSE.

## Prediction

After training, a prediction function standardizes new input using the **same training mean and standard deviation**:

```python
new_person_scaled = (
    new_person - X_mean
) / X_std
```

The trained model then calculates:

```python
prediction = new_person_scaled @ w + b
```

For the test example:

```python
test_height = 175
test_age = 25
```

the model produces a predicted mass based on the parameters learned during training.

## Technologies Used

* Python
* NumPy
* Jupyter Notebook

No Scikit-Learn was used.

No Pandas was used.

The main goal was to implement the fundamental mechanics of linear regression manually with NumPy.

## Project Structure

```text
BodyWeightModel/
│
├── BodyWeightModel.ipynb
└── README.md
```

## What I Learned

This project helped me understand the fundamentals behind a basic machine learning training loop:

```text
Data
 ↓
Features
 ↓
Standardization
 ↓
Prediction
 ↓
Error
 ↓
MSE
 ↓
Gradients
 ↓
Parameter Update
 ↓
Repeat
```

Instead of treating machine learning as a black box, I wanted to understand the mathematical operations happening underneath a simple regression model.

## Repository

GitHub:

https://github.com/AfhamAI/BodyWeightModel

## Note

This is a **learning project** created to understand the fundamentals of implementing linear regression and gradient descent with NumPy.

The model is intentionally simple and should not be considered a scientifically validated method for estimating an individual's actual body weight.

---

**Built from scratch with Python + NumPy.**
