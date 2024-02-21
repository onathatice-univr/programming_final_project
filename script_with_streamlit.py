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
import function as fc

def main():

    st.title("STUDENTS EXAM SCORE ANALYSIS")
    st.markdown("Data science project presentation!")
    st.write("## Introduction")
    st.write("Our project aims to analyze the performance of students based on various factors.")


    ## SIDEBAR ##
    st.sidebar.title('Students Exam Score Anlysis')
    session_selection = st.sidebar.radio(
        "Sessions",
        ("Data Exploration", "Dataset cleaning", "Plots", "Model")
    )
    df_exam_score = pd.read_csv("Expanded_data_with_more_features.csv")
    pd.options.display.float_format = '{:,.2f}'.format
    
    df_exam_score.drop('Unnamed: 0',axis=1, inplace=True)
    df_exam_score_copy = df_exam_score.copy()
    # print out the unique values for each of columns
    fc.print_unique_values(df_exam_score[['EthnicGroup','ParentEduc','Gender','LunchType','TestPrep','ParentMaritalStatus','PracticeSport','IsFirstChild','NrSiblings','TransportMeans','WklyStudyHours']])
    # Cleaning parent education by replacing specific values with a more general value. 
    df_exam_score.loc[df_exam_score['ParentEduc'] == 'some high school', 'ParentEduc'] = 'high school'
    df_exam_score.loc[df_exam_score['ParentEduc'] == 'some college', 'ParentEduc'] = "bachelor's degree"

####### NUMERICAL ENCODING FOR EACH VARIABLE #######
    # ParentEduc: Numerical encoding from 0 to 3 for different education levels (0 is lower and 3 is higher) #######
    replace_parentEduc={
            'high school':0,
            "bachelor's degree":1,
            "master's degree":2,
            "associate's degree":3
        }
    df_exam_score['ParentEduc'].replace(replace_parentEduc,inplace=True)

    # Gender: Binary encoding: 1 for female, 0 for male.
    replace_gender={
            'male':0,
            "female":1
        }
    df_exam_score['Gender'].replace(replace_gender,inplace=True)
    
    # LunchType: Binary encoding: 0 for standard, 1 for free/reduced.
    replace_LunchType={
            'standard':0,
            "free/reduced":1
        }
    df_exam_score['LunchType'].replace(replace_LunchType,inplace=True)

    # TestPrep: Binary encoding: 1 for completed course, 0 for none.
    replace_TestPrep={
            'none':0,
            "completed":1
        }
    df_exam_score['TestPrep'].replace(replace_TestPrep,inplace=True)

    # PracticeSport: Numerical encoding: 0 for never, 1 for sometimes, 2 for regularly.
    replace_PracticeSport={
            'never':0,
            "sometimes":1,
            "regularly":2
        }
    df_exam_score['PracticeSport'].replace(replace_PracticeSport,inplace=True)
        
    # IsFirstChild: Binary encoding: 1 for yes, 0 for no.
    replace_IsFirstChild={
            'no':0,
            "yes":1
        }
    df_exam_score['IsFirstChild'].replace(replace_IsFirstChild,inplace=True)
        
    # TransportMeans: Binary encoding: 0 for schoolbus, 1 for private transport.
    replace_TransportMeans={
            'school_bus':0,
            "private":1
        }
    df_exam_score['TransportMeans'].replace(replace_TransportMeans,inplace=True)
       
    # WklyStudyHours: Numerical encoding: 0 for <5 hours, 1 for 5-10 hours, 2 for >10 hours.
    replace_WklyStudyHours={
            '< 5' :0,
            '5 - 10':1,
            '> 10':2
        }
    df_exam_score['WklyStudyHours'].replace(replace_WklyStudyHours, inplace=True)

    # IMPUTING MISSING VALUES WITH RANDOM FOREST
    for column in df_exam_score.columns:
        if df_exam_score[column].isna().any():
            df_exam_score = fc.impute_missing_values_with_random_forest(df_exam_score, column)
    
    # EthnicGroup: One-hot encoding: Each ethnic group has a different binary variable.
    df_exam_score = pd.get_dummies(df_exam_score, columns=['EthnicGroup', 'ParentMaritalStatus'], dummy_na=True)

    df_exam_score.drop('EthnicGroup_nan', axis=1, inplace=True)
    df_exam_score.drop('ParentMaritalStatus_nan', axis=1, inplace=True)

#_______________###############_______________
    if session_selection == "Data Exploration":

        st.write("# Data Exploration")
        st.write("## Dataset Overview")
        st.write("Check out [dataset](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores?select=Expanded_data_with_more_features.csv)")
        
        # Displaying dataset overview
        with st.expander("More info about Features"):
            features = [
                "Gender: Gender of the student (male/female)",
                "EthnicGroup: Ethnic group of the student (group A to E)",
                "ParentEduc: Parent(s) education background (from some_highschool to master's degree)",
                "LunchType: School lunch type (standard or free/reduced)",
                "TestPrep: Test preparation course followed (completed or none)",
                "ParentMaritalStatus: Parent(s) marital status (married/single/widowed/divorced)",
                "PracticeSport: How often the student practices sport (never/sometimes/regularly)",
                "IsFirstChild: If the child is the first child in the family or not (yes/no)",
                "NrSiblings: Number of siblings the student has (0 to 7)",
                "TransportMeans: Means of transport to school (schoolbus/private)",
                "WklyStudyHours: Weekly self-study hours (less than 5hrs; between 5 and 10hrs; more than 10hrs)",
                "MathScore: math test score (0-100)",
                "ReadingScore: reading test score (0-100)",
                "WritingScore: writing test score (0-100)"
            ]
            for feature in features:
                st.write(feature)

        st.write("## Data Cleaning")
        st.dataframe(df_exam_score.head())

#_______________###############_______________
    elif session_selection == "Dataset cleaning":
        st.write("Dataset cleaning")
        st.write("## HANDLING MISSING VALUES")
        st.dataframe(df_exam_score.head())

#_______________###############_______________
    elif session_selection == "Plots":

        st.write("## Correlation")
        df_exam_score.WklyStudyHours.unique()
        df_exam_score_corr = df_exam_score.corr()
        st.dataframe(df_exam_score_corr)

        fig, ax = plt.subplots(figsize=(18, 14))
        sns.heatmap(df_exam_score_corr, annot=True, ax=ax)
        st.pyplot(fig)

#_______________###############_______________
    elif session_selection == "Model":
        st.write("Model")


if __name__ == "__main__":
    main()
