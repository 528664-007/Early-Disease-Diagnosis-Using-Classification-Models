import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Expanded dataset with 50 diseases
data = {
    "Fever": [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    "Cough": [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    "Headache": [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1],
    "Fatigue": [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    "Chest Pain": [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    "Nausea": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    "Disease": [
        "Flu", "Cold", "Migraine", "Flu", "Anemia", "Pneumonia", "Dengue", "Malaria",
        "Covid-19", "Tuberculosis", "Asthma", "Sinusitis", "Bronchitis", "Typhoid", "Cholera",
        "Hepatitis A", "Hepatitis B", "Hepatitis C", "Strep Throat", "Measles",
        "Mumps", "Chickenpox", "Malaria", "Ebola", "Zika Virus", "Rabies", "Swine Flu",
        "Yellow Fever", "Lyme Disease", "Tetanus", "Whooping Cough", "Scarlet Fever",
        "Rheumatic Fever", "Shingles", "Gonorrhea", "Syphilis", "Chikungunya", "H1N1",
        "Anthrax", "Diphtheria", "Plague", "Smallpox", "Polio", "Meningitis",
        "Encephalitis", "Celiac Disease", "Epilepsy", "Parkinson's Disease", "Alzheimer's", "ALS"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Separating features and target variable
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization
model = RandomForestClassifier(random_state=42)

# Training the model
model.fit(X_train, y_train)

# Predictions for testing
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to predict disease based on user input
def predict_disease(symptoms):
    symptom_vector = np.array([symptoms])  # Convert input to NumPy array
    prediction = model.predict(symptom_vector)
    return prediction[0]

# Extract symptom columns
symptom_columns = X.columns.tolist()

# Repeatedly prompt the user for input until they choose to stop
while True:
    print("\nEnter the following symptoms (1 for Yes, 0 for No):")
    user_symptoms = []
    for symptom in symptom_columns:  # Loop through all symptom columns
        while True:
            try:
                value = int(input(f"{symptom}: "))
                if value in [0, 1]:
                    user_symptoms.append(value)
                    break
                else:
                    print("Please enter 1 or 0 only.")
            except ValueError:
                print("Invalid input. Please enter 1 or 0.")

    # Predict disease
    predicted_disease = predict_disease(user_symptoms)
    print(f"\nPredicted Disease: {predicted_disease}")

    # Ask the user if they want to continue or exit
    continue_choice = input("\nDo you want to predict another disease? (yes/no): ").strip().lower()
    if continue_choice != "yes0":
        print("Exiting the program. Take care!")
        break
