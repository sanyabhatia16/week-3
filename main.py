import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("heart.csv")

# Data Exploration
print("Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Class Distribution:\n", df["target"].value_counts(normalize=True))

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Function to cap outliers using IQR method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)

# Handle outliers for key numerical columns
for col in ["chol", "trestbps", "oldpeak", "thalach"]:
    cap_outliers(df, col)

# Normalize numerical features
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# One-hot encode categorical features
categorical_features = ["cp", "restecg", "slope", "thal"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Split dataset into features (X) and target (y)
X = df.drop(columns=["target"])
y = df["target"]

# Reshape data for CNN: Convert (samples, features) → (samples, features, 1)
X_reshaped = np.expand_dims(X.values, axis=2)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42, stratify=y)

# Define the ANN + CNN Model
model = keras.Sequential([
    # CNN Layers
    layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),

    # ANN Layers
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
