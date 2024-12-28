# Titanic-Survival-Prediction  

---

## Project Overview  
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. By leveraging a combination of feature engineering, handling class imbalances, and model evaluation, the project demonstrates the end-to-end process of building a robust predictive model.

---

## Approach  

1. **Data Preprocessing**  
   - Loaded the Titanic dataset.  
   - Addressed missing values in critical features like `Age` and `Fare`. Missing values were imputed using the median to maintain data consistency.  
   - Encoded categorical variables such as `Sex` and `Embarked` into numerical formats for model compatibility.  

2. **Feature Scaling**  
   - Used `StandardScaler` to normalize continuous features (`Age` and `Fare`) to improve model performance by ensuring equal weighting of features.  

3. **Handling Class Imbalance**  
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and prevent the model from being biased toward the majority class.  

4. **Model Training**  
   - Split the dataset into training (80%) and testing (20%) sets.  
   - Used a **Random Forest Classifier** for training due to its ability to handle both classification tasks and feature importance.  
   - Evaluated the model on the test set, achieving **92% accuracy** with high precision and recall values.  

5. **Model Evaluation**  
   - Saved the trained model as a `.pkl` file using `joblib` for reproducibility.  
   - Evaluated the model using accuracy, classification report, and confusion matrix to provide comprehensive performance metrics.  

---

## Methods Used  

1. **Data Preprocessing**  
   - Handled missing data and encoded categorical variables.  
   - Scaled features for better model convergence.  

2. **Imbalanced Data Handling**  
   - SMOTE was implemented to oversample the minority class, ensuring balanced training.  

3. **Model Selection**  
   - Random Forest Classifier was chosen for its robustness and interpretability.  

4. **Model Saving**  
   - Trained model and scaler were saved to enable easy evaluation without retraining.  

---

## Challenges Faced  

1. **Handling Missing Data**  
   - Missing values in key features like `Age` and `Fare` required careful imputation.  
2. **Class Imbalance**  
   - The dataset exhibited a significant imbalance in survival classes, leading to biased initial predictions. This was addressed effectively using SMOTE.  
3. **Model Performance**  
   - Ensuring that the model generalized well across unseen data required iterative evaluation and hyperparameter tuning.  

---

## Model Performance  

1. **Evaluation Metrics**  
   - **Accuracy**: 92%  
   - **Precision & Recall**: Demonstrated strong precision and recall for both classes, highlighting the model's reliability.  

2. **Confusion Matrix**  
              precision    recall  f1-score   support  
      0       0.94      0.91      0.93       105  
      1       0.88      0.92      0.90        74  

  accuracy                           0.92       179  
 macro avg       0.91      0.92      0.91       179  


---

#### **How to Run**  

1. **Train the Model**  
- Run `train.py` to preprocess the data, train the model, and save the trained model to a file.  

2. **Evaluate the Model**  
- Run `evaluate.py` to load the trained model, preprocess the test set, and evaluate model performance using metrics like accuracy and confusion matrix.  

---

#### **Summary**  
The project successfully demonstrates a complete machine learning pipeline to predict Titanic survival. By addressing class imbalances and scaling features, the Random Forest model achieves high accuracy and reliability. This project highlights the importance of data preprocessing, model evaluation, and balancing techniques in building effective machine learning models.  
