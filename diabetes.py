import streamlit as st
import pickle
import numpy as np


# Load the trained model
# model = pickle.load(open('diabetes_dataset.sav','rb'))
# filename = 'diabetes_dataset.sav'
# with open(filename, 'rb') as file:
#     model = pickle.load(file)
model_RF = pickle.load(open('diabetes_dataset.sav', 'rb'))

# Function to make predictions
def predict_diabetes(features):
    prediction = model_RF.predict([features])
    return prediction[0]
    
# Streamlit app
st.title("Diabetes Classification App")

st.write("""
Masukkan data pasien untuk mengetahui apakah mereka menderita diabetes atau tidak.
""")

# Input features dictionary
input_features = {
    'RegularMedicine': 'Regular Medicine',
    'Age': 'Age',
    'BMI': 'BMI',
    'Family_Diabetes': 'Family Diabetes',
    'Stress': 'Stress',
    'HighBP': 'HighBP',
    'SoundSleep': 'SoundSleep',
    'PhysicallyActive': 'Physically Active',
    'Sleep': 'Sleep',
    'BPLevel': 'BP Level'
}

# Process input features
features = {}
for key, label in input_features.items():
    features[key] = st.text_input(label)

    # Handle empty inputs
    if features[key] == '':
        features[key] = 0
    elif key in ['Age', 'BMI', 'SoundSleep', 'Sleep']:
        features[key] = float(features[key])

# Collect user input into a feature array
features = np.array([features[key] for key in input_features])

# Prediction
if st.button('Klasifikasi'):
    result = predict_diabetes(features)
    
    if result == 1:
        st.error('Pasien menderita diabetes.')
    else:
        st.success('Pasien tidak menderita diabetes.')
