from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'iris_model.pkl')

print("✅ Model trained and saved as 'iris_model.pkl'")
