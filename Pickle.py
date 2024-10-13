import pickle
from sklearn.ensemble import RandomForestClassifier

# Example: Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
with open('model_pickle.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model from a file
with open('model_pickle.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
