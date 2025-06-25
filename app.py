import streamlit as st
import joblib

loaded_model = joblib.load("linear_regression_model.pkl")
loaded_scaler = joblib.load('scaler.pkl')

st.title("Students Test Score Prediction")
st.write("Enter the number of hours the student studied to predict the test score")

hours = st.number_input("Hours Studied:", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        data = [[hours]]
        scaled_data = loaded_scaler.transform(data)
        prediction = loaded_model.predict(scaled_data)
        st.write("Predicted Test Score:", prediction[0])
    except Exception as e:
        st.error(f"Error: {e}")
