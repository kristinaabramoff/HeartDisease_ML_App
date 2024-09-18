import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

# Determine the directory where the .py file is located and assumes models are in a subfolder called models
models_dir = Path(__file__).parent / "models"

# Load the regression model (adjust the path)
filename = models_dir / 'logistic_regression_model.sav'  
loaded_model = pickle.load(open(filename, 'rb'))

# Load the XGBoost model (adjust the path)
filename_xgboost = models_dir / 'xgboost_model.sav'  
loaded_model_xgboost = pickle.load(open(filename_xgboost, 'rb'))

# Load the random forest model (adjust the path)
filename_randomforest = models_dir / 'random_forest_model.sav'  
loaded_model_randomforest = pickle.load(open(filename_randomforest, 'rb'))

# Load the neural networks model (adjust the path)
filename_neuralnetworks = models_dir / 'nn.sav' 
loaded_model_neuralnetworks = pickle.load(open(filename_neuralnetworks, 'rb'))

# Add a title to the app
st.title('Heart Disease Prediction App')

# Add a bold welcome text at the top
st.markdown("**Group 4 Project**")

# Collect the data for features
age = st.slider("Age", min_value=0, max_value=120, value=40)
resting_bp = st.slider("Resting Blood Pressure", min_value=0, max_value=300, value=120)
cholesterol = st.slider("Cholesterol Level", min_value=0, max_value=500, value=200)
max_hr = st.slider("Maximum Heart Rate", min_value=0, max_value=250, value=150)
oldpeak = st.number_input("Enter Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=0.0)
fasting_bs = st.number_input("Enter Fasting Blood Sugar (1 if >120 mg/dl, 0 otherwise)", min_value=0, max_value=1, value=0)

# Collect categorical input features with automatic setting of related features
sex = st.radio("Select Sex", options=["Male", "Female"])
sex_m = sex == "Male"
sex_f = not sex_m

#ChestPainType_ASY	ChestPainType_ATA	ChestPainType_NAP	ChestPainType_TA
chest_pain_type = st.selectbox("Select Chest Pain Type", options=["ASY - Asymptomatic", "ATA - Atypical Angina", "NAP - Non-Anginal Pain", "TA - Typical Angina"])
chest_pain_asy = chest_pain_type.startswith("ASY")
chest_pain_ata = chest_pain_type.startswith("ATA")
chest_pain_nap = chest_pain_type.startswith("NAP")
chest_pain_ta = chest_pain_type.startswith("TA")

resting_ecg = st.selectbox("Select Resting ECG Type", options=["LVH - left ventricular hypertrophy", "Normal - Normal", "ST - having ST-T wave abnormality"])
resting_ecg_lvh = resting_ecg_normal = resting_ecg_st = False
if resting_ecg == "LVH":
    resting_ecg_lvh = True
elif resting_ecg == "Normal":
    resting_ecg_normal = True
elif resting_ecg == "ST":
    resting_ecg_st = True

exercise_angina = st.radio("Do you have Exercise-Induced Angina?", options=["Yes", "No"])
exercise_angina_y = exercise_angina_n = False
if exercise_angina == "Yes":
    exercise_angina_y = True
    exercise_angina_n = False
else:
    exercise_angina_y = False
    exercise_angina_n = True

st_slope = st.selectbox("Select ST Slope Type", options=["Up - upsloping", "Flat - flat", "Down - downsloping"])
st_slope_up = st_slope_flat = st_slope_down = False
if st_slope == "Up":
    st_slope_up = True
elif st_slope == "Flat":
    st_slope_flat = True
elif st_slope == "Down":
    st_slope_down = True

# Button to make prediction
if st.button('Predict'):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([[age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak, sex_f, sex_m, chest_pain_asy, chest_pain_ata, chest_pain_nap, 
                                chest_pain_ta, resting_ecg_lvh, resting_ecg_normal, resting_ecg_st, exercise_angina_n, exercise_angina_y, st_slope_down, 
                                st_slope_flat, st_slope_up]],
                              columns=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 
                                       'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST', 
                                       'ExerciseAngina_N', 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'])
    
    
    #To Test for Heart Disease
    #input_data = pd.DataFrame([[38, 110, 289, 0, 105, 1.5, False, True, True, False, False, 
    #                            False, False, True, False, False, True, True, 
    #                            False, False]])
    

    # Display summary of features data
    st.write("Summary table of features input data")
    st.write(input_data)  

     # Make predictions using the models
    prediction_regression = int(loaded_model.predict(input_data)[0])
    prediction_xgboost = int(loaded_model_xgboost.predict(input_data)[0])
    prediction_randomforest = int(loaded_model_randomforest.predict(input_data)[0])
    prediction_neuralnetworks = int(loaded_model_neuralnetworks.predict(input_data)[0])
    
    # Prepare the prediction results for display in a table
    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'Random Forest' , 'Neural Network'],
        'Prediction': ['Risk of Heart Disease' if x == 1 else 'No Risk of Heart Disease' for x in [prediction_regression, prediction_xgboost, prediction_randomforest, prediction_neuralnetworks]]
    })

    # Display the prediction results in a table
    st.markdown("### Prediction Results:")
    st.table(results)