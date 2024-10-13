import os
import joblib
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Create a directory to save models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset and train model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Generate a timestamp for versioning
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"{MODEL_DIR}/rf_model_joblib_{timestamp}.pkl"

# Metadata dictionary for the model
model_metadata = {
    "model_type": "RandomForestClassifier",
    "train_timestamp": timestamp,
    "accuracy": model.score(X_test, y_test)
}

# Save the model and metadata
try:
    joblib.dump({"model": model, "metadata": model_metadata}, model_filename)
    print(f"Model saved successfully using joblib to {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

# Load the model and metadata
try:
    loaded_data = joblib.load(model_filename)
    loaded_model = loaded_data["model"]
    loaded_metadata = loaded_data["metadata"]
    print(f"Model loaded successfully using joblib.")
    print(f"Model metadata: {loaded_metadata}")
except Exception as e:
    print(f"Error loading the model: {e}")

# Test the loaded model
accuracy_joblib = loaded_model.score(X_test, y_test)
print(f"Accuracy of the loaded model (joblib): {accuracy_joblib}")
