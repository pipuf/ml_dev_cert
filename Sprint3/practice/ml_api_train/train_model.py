import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score 

# Load the dataset
print(f'Loading the dataset...')
url = 'Dataset/diabetes.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(f'Dataset:')
print(data.head())

print(f'Preprocessing the dataset...')
# Define the features and target
X = data.drop(columns=["diabetes"])
y = data["diabetes"]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column transformations
numeric_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
numeric_transformer = StandardScaler()

categorical_features = ["gender", "smoking_history"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Create the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)),
    ]
)

print(f'Training the model...')
# Train the model
model.fit(X_train, y_train)

print(f'Testing the model...')
# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the preprocessor and the model separately for API usage
print(f'Saving the model and preprocessor...')
# Extract and save the preprocessor
with open("assets/preprocessor.pkl", "wb") as f:
    pickle.dump(model.named_steps["preprocessor"], f)

# Save the model
with open("assets/model.pkl", "wb") as f:
    pickle.dump(model.named_steps["classifier"], f)

print('Model and preprocessor saved successfully!')