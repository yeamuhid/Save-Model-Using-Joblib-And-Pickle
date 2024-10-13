import joblib
from sklearn.ensemble import RandomForestClassifier

# Example: Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'model_joblib.pkl')

# Load the model from a file
loaded_model = joblib.load('model_joblib.pkl')
