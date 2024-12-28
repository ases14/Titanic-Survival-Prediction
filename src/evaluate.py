import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_file, data_file):
    try:
        # Load the model
        model = joblib.load(model_file)
        print(f"Loaded model type: {type(model)}")

        # Check if the model has a 'predict' method
        if not hasattr(model, 'predict'):
            raise TypeError("The loaded model is not a valid machine learning model with a 'predict' method.")
        
        # Load the data
        data = pd.read_csv(data_file)
        X = data.drop('Survived', axis=1)
        y = data['Survived']

        # Split data into train and test sets (same split as in train.py)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale 'Age' and 'Fare' columns using the same scaler that was fit on the training data
        scaler = StandardScaler()
        X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
        X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

        # Predict on the test data using the loaded model
        y_pred = model.predict(X_test)

        # Evaluate the performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on evaluation set: {accuracy:.2f}")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # Plotting confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise

if __name__ == "__main__":
    model_file = 'C:/Code/Titanic-Survival-Prediction/models/titanic_model.pkl'  # Path to the saved model
    data_file = 'C:/Code/Titanic-Survival-Prediction/data/preprocessed_titanic.csv'  # Path to the preprocessed data
    evaluate_model(model_file, data_file)
