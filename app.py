import streamlit as st
import joblib
import numpy as np

models = joblib.load("All_model.joblib")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo_znu.jpeg", width=200)

st.title("Student Performance Prediction")
st.write("Enter Student`s Data and Exam_Score will be predicted.")

# select model
model_names = list(models.keys())
Select_model = st.selectbox("Choose Model: ", model_names)
best_model = models[Select_model]
st.write("select model: ", Select_model)

# Inputs For User
Hours_Studied = st.number_input("Hours Studied: ", min_value=1, max_value=45, value=12)
Attendance = st.number_input("Attendance: ", min_value=55, max_value=100, value=70)
Study_Efficiency = st.number_input("Study Efficiency: ", min_value=1, max_value=38, value=12)
Study_Intensity = st.number_input("Study Intensity: ", min_value=0, max_value=10, value=5)
Academic_Pressure = st.number_input("Academic Pressure: ", min_value=-9, max_value=40, value=10)
Study_Support = st.number_input("Study Support: ", min_value=1, max_value=55, value=20)
Previous_Scores = st.number_input("Previous Scores: ", min_value=50, max_value=100, value=80)
Tutoring_Sessions = st.number_input("Tutoring Sessions: ", min_value=0, max_value=7, value=5)
Parental_Involvement_Low = st.number_input("Parental Involvement Low (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Parental_Involvement_Medium = st.number_input("Parental Involvement Medium (Yes=>1, No=>0): ", min_value=0, max_value=1, value=1)
Access_to_Resources_Low = st.number_input("Access to Resources Low (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Extracurricular_Activities_Yes = st.number_input("Extracurricular Activities Yes (Yes=>1, No=>0): ", min_value=0, max_value=1, value=1)
Motivation_Level_Low = st.number_input("Motivation Level Low (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Family_Income_Low = st.number_input("Family Income Low (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Internet_Access_Yes = st.number_input("Internet Access Yes (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Teacher_Quality_Low = st.number_input("Teacher Quality Low (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Teacher_Quality_Medium = st.number_input("Teacher Quality medium (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Peer_Influence_Positive = st.number_input("Peer Influence Positive (Yes=>1, No=>0): ", min_value=0, max_value=1, value=1)
Learning_Disabilities_Yes = st.number_input("Learning Disabilities Yes (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Parental_Education_Level_High_School = st.number_input("Parental Education Level High School (Yes=>1, No=>0): ", min_value=0, max_value=1, value=1)
Parental_Education_Level_Postgraduate = st.number_input("Parental Education Level Postgraduate (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)
Distance_from_Home_Moderate = st.number_input("Distance from Home Moderate (Yes=>1, No=>0): ", min_value=0, max_value=1, value=1)
Distance_from_Home_Near = st.number_input("Distance from Home Near (Yes=>1, No=>0): ", min_value=0, max_value=1, value=0)

# predict
if st.button("Predict Exam_Score"):
    features = np.array([[
        Hours_Studied, Attendance, Previous_Scores, Tutoring_Sessions,
        Study_Efficiency, Study_Intensity,Academic_Pressure, Study_Support,Parental_Involvement_Low,
        Parental_Involvement_Medium, Access_to_Resources_Low,
        Extracurricular_Activities_Yes, Motivation_Level_Low,
        Internet_Access_Yes, Family_Income_Low, Teacher_Quality_Low,
        Teacher_Quality_Medium, Peer_Influence_Positive,
        Learning_Disabilities_Yes, Parental_Education_Level_High_School,
        Parental_Education_Level_Postgraduate, Distance_from_Home_Moderate,
        Distance_from_Home_Near

    ]])

    prediction = best_model.predict(features)
    st.success(f"Exam_Score:  {prediction[0]}")
