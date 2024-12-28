# Data Preprocessing Steps  

This document explains the preprocessing steps applied to the Titanic dataset:  

1. **Handling Missing Values**:  
   - Dropped rows with missing values in less significant features.  
   - Imputed missing `Age` values using the median.  

2. **Feature Scaling**:  
   - Applied `StandardScaler` to the `Age` and `Fare` columns to normalize them.  

3. **Encoding Categorical Variables**:  
   - Converted categorical columns like `Sex` and `Embarked` into numerical representations using one-hot encoding.  

4. **Class Imbalance**:  
   - Used SMOTE to balance the dataset.  
   - Resulted in a balanced dataset with an equal number of survivors and non-survivors.  
