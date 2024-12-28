import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

def preprocess_data(input_file, output_file):
    try:
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        print("Dropping unnecessary columns...")
        df.drop(columns=['Cabin'], inplace=True)
        df.drop(['Name', 'Ticket'], axis=1, inplace=True)
        
        print("Imputing missing Age values...")
        df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
        
        print("Filling missing Embarked and Fare values...")
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        
        print("Encoding categorical variables...")
        label_encoder = LabelEncoder()
        df['Sex'] = label_encoder.fit_transform(df['Sex'])
        df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
        
        print("Data preprocessing completed successfully.")
        
        # Ensure the directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Saving preprocessed data
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    input_file = 'C:/Code/Titanic-Survival-Prediction/data/titanic.csv'
    output_file = 'C:/Code/Titanic-Survival-Prediction/data/preprocessed_titanic.csv'
    preprocess_data(input_file, output_file)
