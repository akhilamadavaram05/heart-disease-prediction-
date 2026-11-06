import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("heart.csv")

# Separate input features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the machine learning model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Heart Disease Prediction System")
print("--------------------------------")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Test one sample from the test data
sample = X_test.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: Heart disease may be present.")
else:
    print("Prediction: Heart disease may not be present.")
