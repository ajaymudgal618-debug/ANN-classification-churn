import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  , LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle 
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st


## load the model
model = load_model("model.h5")

#  lload the lb
with open("lb_risk.pkl", "rb") as file:
    lb_risk= pickle.load(file)

# load  the scaler
with open("scaler.pkl", "rb")as file:
    scaler= pickle.load(file)


## streamlit setup
st.title("loan risk predictor")
input_data = {
    "Name": st.text_input("enter the name"),
    "Age": st.slider("enter the age", min_value=1, max_value=100),
    "Income": st.number_input("enter the income", min_value=0 , step=5000),
    "Loan": st.number_input("enter the loan amount", min_value=0, step=5000)
}


data = pd.DataFrame([input_data])
data= data.drop(["Name"], axis = 1 )

## scale the data
data_scaled= scaler.transform(data)

## predict
prediction=model.predict(data_scaled)
predict = prediction[0][0]
if predict >= 0.5:
    risk = "low Risk"
else:
    risk="high risk"


st.write(f"risk of granting the loan to the person is {risk}")