import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'model_joblib.pkl')

# Load the model using joblib
loaded_model_joblib = joblib.load('model_joblib.pkl')

# Test the loaded model
accuracy_joblib = loaded_model_joblib.score(X_test, y_test)
print(f"Accuracy of the loaded model (joblib): {accuracy_joblib}")
