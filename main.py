import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
file_path = r"ML Rock or mine\Data.csv"
sonar_data = pd.read_csv(file_path, header=None)

# Splitting data into features and target
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60].map({'R': 0, 'M': 1})  # Convert labels to numeric

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

#Debug: Check if model is trained properly
print(f"Train Accuracy: {model.score(X_train, Y_train):.2f}")
print(f"Test Accuracy: {model.score(X_test, Y_test):.2f}")

#Take user input
user_input = input("Enter 60 sonar values(Rock/Mine) separated by commas(AVL in Data.csv): \n")
print("User input received:", user_input)  # Debugging line

try:
    # Convert input string to a NumPy array
    input_data = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)
    
    #Debug: Check if input is reshaped correctly
    print("Input Data Shape:", input_data.shape)

    # Ensure the input has 60 features
    if input_data.shape[1] != 60:
        print("Error: Please enter exactly 60 values.")
    else:
        # Make a prediction
        prediction = model.predict(input_data)
        print("Prediction Output:", prediction)  # Debugging line

        # Display result
        if prediction[0] == 0:
            print("The object is a Rock 🪨")
        else:
            print("The object is a Mine 💣")

except ValueError:
    print("Error: Please enter valid numerical values separated by commas.")

