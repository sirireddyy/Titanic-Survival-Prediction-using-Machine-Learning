# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\Siri Reddy\OneDrive\Desktop\KODBUD\Task 2\archive (2)\Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill missing Age values with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fill missing Fare values with mean
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Convert categorical columns into numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# -----------------------------
# Feature Selection
# -----------------------------

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

# Target column
y = df['Survived']

# Check for missing values
print("\nMissing values:\n")
print(X.isnull().sum())

# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# -----------------------------
# Visualization
# -----------------------------

survival_counts = df['Survived'].value_counts()

plt.bar(['Not Survived', 'Survived'], survival_counts)

plt.title("Titanic Survival Count")

plt.xlabel("Survival Status")

plt.ylabel("Number of Passengers")

plt.show()