import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def train_model(input_file, model_file):
    try:
        # Load the preprocessed data
        print(f"Loading preprocessed data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Separate features and target variable
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Resampled data size: {X_resampled.shape}")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Scale 'Age' and 'Fare' features
        scaler = StandardScaler()
        X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
        X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])
        
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        print("Training the model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        print("Evaluating the model...")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the trained model
        joblib.dump(model, model_file)
        print(f"Model saved to {model_file}")
    
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    input_file = 'C:/Code/Titanic-Survival-Prediction/data/preprocessed_titanic.csv'  # Path to the preprocessed data
    model_file = 'C:/Code/Titanic-Survival-Prediction/models/titanic_model.pkl'  # Path to save the trained model
    train_model(input_file, model_file)
