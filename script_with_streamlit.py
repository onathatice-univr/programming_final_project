import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
from scipy import stats


def main():

    ## SIDEBAR ##
    st.sidebar.title('Students Exam Score Anlysis')
    session_selection = st.sidebar.radio(
        "Sessions",
        ("Data Exploration", "Dataset cleaning", "Plots", "Model")
    )
    
    if session_selection == "Data Exploration":
        st.write("Data Exploration")

    elif session_selection == "Dataset cleaning":
        st.write("Dataset cleaning")

    elif session_selection == "Plots":
        st.write("Plots")
    
    elif session_selection == "Model":
        st.write("Model")


    ## CONTENT ##
    st.title("FINAL PROJECT: STUDENTS EXAM SCORE")
    st.markdown("Data science project presentation!")

    st.write("## Introduction")
    st.write("Our project aims to analyze the performance of students based on various factors.")

    st.write("## Dataset Overview")
    st.write("Check out [dataset](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores?select=Expanded_data_with_more_features.csv)")
    st.write("Here is a brief overview of the dataset:")
    
    # Displaying dataset overview
    st.write("1. Gender: Gender of the student (male/female)")
    st.write("2. EthnicGroup: Ethnic group of the student (group A to E)")
    st.write("3. ParentEduc: Parent(s) education background (from some_highschool to master's degree)")
    st.write("4. LunchType: School lunch type (standard or free/reduced)")
    st.write("5. TestPrep: Test preparation course followed (completed or none)")
    st.write("6. ParentMaritalStatus: Parent(s) marital status (married/single/widowed/divorced)")
    st.write("7. PracticeSport: How often the student practices sport (never/sometimes/regularly)")
    st.write("8. IsFirstChild: If the child is the first child in the family or not (yes/no)")
    st.write("9. NrSiblings: Number of siblings the student has (0 to 7)")
    st.write("10. TransportMeans: Means of transport to school (schoolbus/private)")
    st.write("11. WklyStudyHours: Weekly self-study hours (less than 5hrs; between 5 and 10hrs; more than 10hrs)")
    st.write("12. MathScore: math test score (0-100)")
    st.write("13. ReadingScore: reading test score (0-100)")
    st.write("14. WritingScore: writing test score (0-100)")

    st.write("## Data Exploration")
    st.write("Data exploration and preprocessing to prepare the dataset for analysis.")
    df_exam_score = pd.read_csv("/home/hatice/Desktop/DS_UniVR/first_year/programmingAndDB/programming_final_project/Expanded_data_with_more_features.csv")
    pd.options.display.float_format = '{:,.2f}'.format 
    df_exam_score.head()
    st.pyplot(df_exam_score.hist())


    st.write("## Correlation Analysis")
    st.write("We analyzed the correlations between different variables in the dataset.")

    st.write("## Model Building")
    st.write("We built machine learning models to predict student performance based on the available features.")

    st.write("## Conclusion")
    st.write("In conclusion, our analysis provides insights into factors influencing student performance.")

if __name__ == "__main__":
    main()
