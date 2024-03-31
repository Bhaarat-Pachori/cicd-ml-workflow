import streamlit as st
from joblib import load
import pandas as pd
from model import le


# Load your trained model
loaded_model = load('model.joblib')
loaded_le = load('le.joblib')


# Define a function to make predictions
def make_prediction(data):
    global loaded_model
    # Perform any preprocessing on the input data if needed
    # Make predictions with your model
    # prediction = model.predict(data)
    prediction_result = loaded_model.predict(pd.DataFrame(data))
    return prediction_result


def prepare_df(data):
    global loaded_le
    single_data = {
        'Gender': ['Male'],
        'Age': [24.443011],
        'Height': [1.699998],
        'Weight': [81.66995],
        'family_history_with_overweight': ['yes'],
        'FAVC': ['yes'],
        'FCVC': [2.0],
        'NCP': [2.983297],
        'CAEC': ['Sometimes'],
        'SMOKE': ['no'],
        'CH2O': [2.763573],
        'SCC': ['no'],
        'FAF': [0.0],
        'TUE': [0.976473],
        'CALC': ['Sometimes'],
        'MTRANS': ['Public_Transportation']
    }
    df = pd.DataFrame(single_data)
    # for item, idx in enumerate(data):
    #     data[idx] = loaded_le.fit_transform([item])
    # return data
    for column in df.columns:
        # Check if the column dtype is object
        if df[column].dtype == 'object':
            df[column] = loaded_le.fit_transform(df[column])
            df[column] = df[column].astype(float)
    return df


# Create the web app interface
st.title('Obesity Prediction App')

st.sidebar.title('Set Input Parameters')
# Add input fields for each feature
# Gender,Age,Height,Weight,family_history_with_overweight,FAVC,FCVC,NCP,CAEC,SMOKE,CH2O,SCC,FAF,TUE,CALC,MTRANS
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
age = st.sidebar.number_input('Age', value=24)
ht = st.sidebar.number_input('Height', value=1.7)
wt = st.sidebar.number_input('Weight', value=1.7)
fhwow = st.sidebar.selectbox("Family History overweight", ("Yes", "No"))
favc = st.sidebar.selectbox("FAVC", ("Yes", "No"))
fcvc = st.sidebar.number_input('FCVC', value=1.7)
ncp = st.sidebar.number_input('NCP', value=1.7)
caec = st.sidebar.selectbox("CAEC", ("Sometimes", "Frequently", "No", "Always"))
smk = st.sidebar.selectbox("Smoke", ("Yes", "No"))
ch2o = st.sidebar.number_input('C2HO', value=1.7)
scc = st.sidebar.selectbox("SCC", ("Yes", "No"))
faf = st.sidebar.number_input('FAF', value=1.7)
tue = st.sidebar.number_input('TUE', value=1.7)
calc = st.sidebar.selectbox("CALC", ["Sometimes", "no", "Frequently"])
mtrans = st.sidebar.selectbox('MTRANS', ('Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'))

# When the user clicks the 'Predict' button
if st.sidebar.button('Predict'):
    # Prepare the input data
    input_data = [[gender, age, ht, wt, fhwow, favc, fcvc, ncp, caec, smk,
                   ch2o, scc, faf, tue, calc, mtrans]]
    # Make predictions
    user_input = pd.DataFrame(input_data)
    user_input = prepare_df(user_input)
    prediction = make_prediction(user_input)
    st.write('Prediction:', prediction)
