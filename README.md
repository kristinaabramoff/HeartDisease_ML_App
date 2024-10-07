# Predicting Heart Disease: A Machine Learning Approach

## Project Overview
This project focuses on predicting the likelihood of heart disease in patients using various machine learning models. The objective is to create an accurate prediction model based on key clinical features such as age, blood pressure, cholesterol levels, and more. This project was developed in collaboration with my teammates, but I personally refined the README and edited the app deployment.

## Features Used for Prediction:
- Age
- Sex
- ChestPainType
- RestingBP
- Cholesterol
- FastingBS
- RestingECG
- MaxHR
- ExerciseAngina
- Oldpeak
- ST_Slope
- HeartDisease (target variable)

## Dataset
The dataset used contains clinical features for predicting heart disease events:
- **Total observations**: 1,190
- **Duplicated observations**: 272
- **Final dataset**: 918 observations after removing duplicates

You can find the dataset here:  
[Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com)

## Preprocessing
The dataset was processed using a combination of Python libraries such as Scikit-learn, PySpark, and Matplotlib for visualization. Google Colab was utilized as the hosted Jupyter Notebook service. The primary data manipulation was handled using Pandas. The preprocessing steps involved:
- Splitting the dataset into features (X) and the target variable (y) representing heart disease.
- Using `train_test_split` from Scikit-learn to split the dataset into training and testing sets with a defined `random_state` for reproducibility.
- Determining the number of input features for model training.

## Machine Learning Models
Since this is a binary classification problem, the following machine learning models were applied:

### Logistic Regression
- Trained on labeled data (X_train, y_train) using the 'liblinear' solver with a set `random_state` and a maximum of 200 iterations.
- Model evaluation included a confusion matrix and classification report, achieving an accuracy of **90%**.

### Random Forest
- Evaluated on 230 instances, achieving a balanced performance with an overall accuracy of **89%**, and high precision and recall in detecting heart disease.

### XGBoost
- The XGBoost model achieved an accuracy of **91%**, outperforming other models. However, after dropping the two least important features, accuracy decreased slightly to **86%**. Due to this drop in performance, the decision was made to retain all features and use the full model, which consistently showed the best results.

### Neural Networks
- Consisted of five hidden layers with decreasing nodes and sigmoid activation functions. The model achieved a final test set accuracy of **89%** after 100 epochs.

## Deployment
The heart disease prediction app was built using **Streamlit**. It incorporates four pre-trained models (Logistic Regression, Random Forest, XGBoost, and Neural Networks) to predict heart disease based on user inputs.

### Features of the App:
- Users input health metrics such as age, cholesterol, and blood pressure.
- The app provides predictions on the likelihood of heart disease using the pre-trained models.
- A user-friendly interface allows for easy interaction and access to predictions.

  
  # Example of app

![app_example](https://github.com/user-attachments/assets/6e868bf0-4c69-4afd-9de4-d1164802b1e3)


## Clinical Relevance of Results

### Healthcare Applications of Predictive Models:
- Machine learning models like Random Forest and XGBoost offer early detection of heart disease, allowing timely interventions for high-risk patients.
- These models reduce the need for costly and invasive procedures by identifying patients who would benefit most from further evaluation.

### Enhancing Healthcare Outcomes:
- Predictive models optimize resource allocation, leading to better care and more accurate diagnoses.

## Conclusion
The project successfully demonstrates the use of machine learning to predict heart disease, with the XGBoost model yielding the best results when all features were retained. The deployed Streamlit app offers an accessible tool for predicting heart disease risk based on clinical data, providing value in early detection and healthcare resource optimization.

## How to Run the Streamlit App

**Install Streamlit** (if not already installed):
   ```bash

   pip install streamlit
Navigate to the project folder:
- cd to HeartDisease_ML_APP folder

- "streamlit run app_1.py" 

This project was done in collaboration with Frank Nguyen, David Rock and Fatema G

