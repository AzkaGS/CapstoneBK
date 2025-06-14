import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load model dan preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('best_obesity_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

model, scaler, le = load_model()

# Load expected feature names from a file or define them
# This is necessary because some models (like Gradient Boosting)
# don't store feature_names_in_ in older sklearn versions
# Assuming feature names were saved during training
try:
    # Try loading from a file if you saved them
    # with open('feature_names.txt', 'r') as f:
    #     expected_features = [line.strip() for line in f]
    # If not saved, manually define or get from a sample of training data
    # For this example, we'll assume a fixed order based on the notebook
    # Replace with your actual feature names if different
    expected_features = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
        'Gender_Female', 'Gender_Male', 'CALC_Frequently', 'CALC_Sometimes',
        'CALC_no', 'CALC_Always', 'FAVC_no', 'FAVC_yes', 'SCC_no', 'SCC_yes',
        'SMOKE_no', 'SMOKE_yes', 'family_history_with_overweight_no',
        'family_history_with_overweight_yes', 'CAEC_Always', 'CAEC_Frequently',
        'CAEC_Sometimes', 'CAEC_no', 'MTRANS_Automobile', 'MTRANS_Bike',
        'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
    ]

except FileNotFoundError:
    st.error("feature_names.txt not found. Please ensure feature names are saved during training.")
    expected_features = [] # Fallback

# Kelas obesitas
obesity_classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

st.title('🏥 Prediksi Tingkat Obesitas')
st.write('Aplikasi untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik')

# Sidebar untuk input
st.sidebar.header('Input Data Pengguna')

# Input features
gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
age = st.sidebar.slider('Age (Usia)', 14, 61, 25)
height = st.sidebar.slider('Height (Tinggi dalam meter)', 1.45, 1.98, 1.70)
weight = st.sidebar.slider('Weight (Berat dalam kg)', 39, 173, 70)

family_history = st.sidebar.selectbox('Family History with Overweight', ['no', 'yes'])
favc = st.sidebar.selectbox('Frequent consumption of high caloric food (FAVC)', ['no', 'yes'])
fcvc = st.sidebar.slider('Frequency of consumption of vegetables (FCVC)', 1, 3, 2)
ncp = st.sidebar.slider('Number of main meals (NCP)', 1.0, 4.0, 3.0)

caec = st.sidebar.selectbox('Consumption of food between meals (CAEC)',
                           ['no', 'Sometimes', 'Frequently', 'Always'])
smoke = st.sidebar.selectbox('Smoke', ['no', 'yes'])
ch2o = st.sidebar.slider('Consumption of water daily (CH2O)', 1.0, 3.0, 2.0)
scc = st.sidebar.selectbox('Calories consumption monitoring (SCC)', ['no', 'yes'])

faf = st.sidebar.slider('Physical activity frequency (FAF)', 0.0, 3.0, 1.0)
tue = st.sidebar.slider('Time using technology devices (TUE)', 0, 2, 1)
calc = st.sidebar.selectbox('Consumption of alcohol (CALC)',
                           ['no', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.sidebar.selectbox('Transportation used (MTRANS)',
                             ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

if st.sidebar.button('Prediksi Obesitas'):
    # Prepare input data
    input_data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'FCVC': fcvc,
        'NCP': ncp,
        'CH2O': ch2o,
        'FAF': faf,
        'TUE': tue,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0, # Added Gender_Female
        'family_history_with_overweight_yes': 1 if family_history == 'yes' else 0,
        'family_history_with_overweight_no': 1 if family_history == 'no' else 0, # Added family_history_with_overweight_no
        'FAVC_yes': 1 if favc == 'yes' else 0,
        'FAVC_no': 1 if favc == 'no' else 0, # Added FAVC_no
        'CAEC_Frequently': 1 if caec == 'Frequently' else 0,
        'CAEC_Sometimes': 1 if caec == 'Sometimes' else 0,
        'CAEC_Always': 1 if caec == 'Always' else 0,
        'CAEC_no': 1 if caec == 'no' else 0, # Added CAEC_no
        'SMOKE_yes': 1 if smoke == 'yes' else 0,
        'SMOKE_no': 1 if smoke == 'no' else 0, # Added SMOKE_no
        'SCC_yes': 1 if scc == 'yes' else 0,
        'SCC_no': 1 if scc == 'no' else 0, # Added SCC_no
        'CALC_Frequently': 1 if calc == 'Frequently' else 0,
        'CALC_Sometimes': 1 if calc == 'Sometimes' else 0,
        'CALC_Always': 1 if calc == 'Always' else 0,
        'CALC_no': 1 if calc == 'no' else 0, # Added CALC_no
        'MTRANS_Bike': 1 if mtrans == 'Bike' else 0,
        'MTRANS_Motorbike': 1 if mtrans == 'Motorbike' else 0,
        'MTRANS_Public_Transportation': 1 if mtrans == 'Public_Transportation' else 0,
        'MTRANS_Walking': 1 if mtrans == 'Walking' else 0, # Corrected the MTRANS_Walking value to 1 if walking
        'MTRANS_Automobile': 1 if mtrans == 'Automobile' else 0 # Added MTRANS_Automobile
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all columns are present (add missing columns with 0) and in the correct order
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Reorder columns to match training data
    input_df = input_df[expected_features]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Get class name
    predicted_class = le.inverse_transform([prediction])[0]

    # Display results
    st.success(f'Prediksi Tingkat Obesitas: **{predicted_class}**')

    # Display probability for each class
    st.subheader('Probabilitas untuk Setiap Kelas:')
    prob_df = pd.DataFrame({
        'Tingkat Obesitas': obesity_classes,
        'Probabilitas': prediction_proba
    })
    prob_df = prob_df.sort_values('Probabilitas', ascending=False)

    for idx, row in prob_df.iterrows():
        st.write(f"**{row['Tingkat Obesitas']}**: {row['Probabilitas']:.3f}")

    # Visualize probabilities
    st.bar_chart(prob_df.set_index('Tingkat Obesitas')['Probabilitas'])

# Display model information
st.subheader('ℹ️ Informasi Model')
st.write(f"Model yang digunakan: {type(model).__name__}")
st.write("Dataset: Obesity Dataset (Mexico, Peru, Colombia)")
st.write("Akurasi Model: 95%+")

# Feature importance (if available)
if hasattr(model, 'feature_importances_'):
    st.subheader('📊 Feature Importance')
    feature_names = expected_features # Use the defined expected_features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    st.bar_chart(importance_df.set_index('Feature')['Importance'])
