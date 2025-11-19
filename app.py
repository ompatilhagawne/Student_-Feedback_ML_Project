import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load Model and Data
# -----------------------------
MODEL_PATH = "/mnt/data/Student_model.pkl"
DATA_PATH = "/mnt/data/Employee_clean_Data.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

model = load_model()
data = load_data()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Student Prediction App")
st.write("This app uses your uploaded Student_model.pkl to make predictions.")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Input Values")

# Dynamically create input fields based on the dataset (except target column)
input_cols = [col for col in data.columns if data[col].dtype != "object"]
inputs = {}

for col in input_cols:
    value = st.number_input(f"Enter {col}", value=float(data[col].mean()))
    inputs[col] = value

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: **{prediction}**")
