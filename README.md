## Project 4 - Predicting Heart Disease
The goal of this project was to use a machine learning model to accurately predict whether a patient will experience heart failure based on the following features:

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
- HeartDisease

## Dataset
The dataset utilized contains 11 clinical features for predicting heart disease events. 

- **Total:** 1,190 observations
- **Duplicated:** 272 observations
- **Final dataset:** 918 observations

[Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com)

## Preprocessing
The Scikit-learn library covered the machine learning workflow from data preprocessing to model evaluation. PySpark was also used to read and inspect the data with Matplotlib being applied for visulisations. The Panda’s library was ultimately used for primary data manipulation with Google Colab utilised for its hosted Juypter Notebook service.

### The Preprocessing Steps Involved
- Splitting the dataset into features (X) and the target variable (y) representing the presence of heart disease.
- The dataset was then divided into training and testing sets using the `train_test_split` method from scikit-learn with a defined `random_state` to ensure reproducibility.
- The number of input features was determined and printed as part of this process.

## Machine Learning Models
Due to this issue being a binary classification problem (predicting whether heart failure occurs or not), the following Machine Learning Models were utilized:

### Logistic Regression
- This model was trained on labeled data (X_train, y_train) using the ‘liblinear’ solver with a set `random_state` for reproducibility and a maximum of 200 iterations.
- Predictions were made on the test set (X_test), and the model's performance was evaluated using a confusion matrix and classification report.
- The confusion matrix revealed a balanced number of true positives (122) and true negatives (84) with some misclassifications (13 false positives and 11 false negatives).
- The classification report showed high precision (0.88-0.90), recall (0.87-0.92), and F1-scores (0.87-0.91) across both classes.
- The model achieved a final accuracy of 90%, indicating reliable performance in predicting heart failure, though minor misclassifications suggest room for further optimization.

### Random Forest
- The Random Forest model was evaluated on 230 instances, classifying healthy (Class 0) and diseased hearts (Class 1).
- It achieved precision, recall, and F1-scores of 0.88, 0.86, and 0.87 for Class 0, and 0.90, 0.92, and 0.91 for Class 1.
- Overall, the model demonstrated an accuracy of 0.89, with macro and weighted averages for precision, recall, and F1-scores consistently at 0.89.
- The model's balanced performance, particularly its high precision and recall in detecting diseased hearts, indicates it is a reliable choice for heart classification tasks.

### XGBoost
- The XGBoost model was tested on 230 instances (Class 0: Healthy Heart, Class 1: Diseased Heart).
- It achieved precision, recall, and F1-scores of 0.90, 0.88, and 0.89 for Class 0, and 0.91, 0.93, and 0.92 for Class 1, with an overall accuracy of 0.91, outperforming other models.
- After removing the three lowest-ranked features, performance slightly declined, with Class 0 metrics at 0.83 precision, 0.86 recall, and 0.84 F1-score, and Class 1 metrics at 0.89 precision, 0.87 recall, and 0.88 F1-score. This indicates that even lower-ranked features contributed to the model’s effectiveness.

### Neural Networks
- Within this model, five hidden layers were applied using a decreasing number of nodes (128 to 16) and sigmoid activation functions.
- Initially utilized `binary_crossentropy` loss, then later switched to `huber` loss, both with the `adam` optimizer.
- Accuracy was the primary evaluation metric, with an accuracy of 75% or higher being optimal.
- The model was trained for 100 epochs with a 15% validation split.
- Training accuracy did improve, but validation accuracy remained stagnant at 49%.
- The model did achieve a final 89% accuracy on the test set, suggesting good performance on the test data.

## Deployment
This project’s deployment is a heart disease prediction app built with Streamlit. It loads four pre-trained machine learning models (Logistic Regression, XGBoost, Random Forest, and Neural Networks) to predict heart disease based on user input.

### Features of the App
- Users can provide various health metrics (age, blood pressure, cholesterol, etc.) through the app's interface.
- The models generate predictions on the likelihood of heart disease.
- The app aims to offer a user-friendly tool for heart disease risk assessment by leveraging multiple machine learning models to enhance prediction accuracy and provide users with insightful health predictions based on their data.

## Clinical Relevance of Results

### Healthcare Applications of Predictive Models
- Predictive models such as Random Forest and XGBoost play a crucial role in the early detection and risk assessment of heart disease.
- These models enable the identification of high-risk patients at an earlier stage, allowing for timely interventions and the development of personalized care plans.

### Minimizing Invasive Procedures
- Invasive screening methods like angiograms and stress tests are often costly and risky.
- By leveraging machine learning models, the need for these procedures can be reduced, focusing instead on patients who are most likely to benefit from further evaluation.

### Enhancing Healthcare Outcomes
- These predictive models enhance healthcare by optimizing resource allocation and improving patient outcomes.
- They provide data-driven insights that lead to more accurate diagnoses and better overall care.
