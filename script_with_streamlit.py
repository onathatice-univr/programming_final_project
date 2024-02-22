import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import metrics
from scipy import stats
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import function as fc

def main():

    st.title("STUDENTS EXAM SCORE ANALYSIS")
    st.write("## Introduction")
    st.write('''The project aims to analyze a dataset containing lifestyle and exam score information about students. 
             Its primary objective is to predict students' scores using regression algorithms. 
             After experimenting with various algorithms, polynomial regression emerged as the most effective method.''')

    df_exam_score = pd.read_csv("exam_score_dataset.csv")
    df_exam_score_copy = df_exam_score.copy()
    pd.options.display.float_format = '{:,.2f}'.format
    
    st.write("# Data Exploration")
    #DATASET, RAW DATA
    st.write("## Dataset Overview")
    st.write("Check out [dataset](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores?select=Expanded_data_with_more_features.csv)")
    with st.expander("More info about Features"):
        features = [
            "Gender: Gender of the student (male=0/female=1)",
            "EthnicGroup: Ethnic group of the student (group A to E, one-hot encoding)",
            "ParentEduc: Parent(s) education background (from some_highschool to master's degree)",
            "LunchType: School lunch type (standard=0 or free/reduced=1)",
            "TestPrep: Test preparation course followed (completed=1 or none=0)",
            "ParentMaritalStatus: Parent(s) marital status (married/single/widowed/divorced, one-hot encoding)",
            "PracticeSport: How often the student practices sport (never=0/sometimes=1/regularly=2)",
            "IsFirstChild: If the child is the first child in the family or not (yes=1/no=0)",
            "NrSiblings: Number of siblings the student has (0 to 7)",
            "TransportMeans: Means of transport to school (schoolbus=0/private=1)",
            "WklyStudyHours: Weekly self-study hours (less than 5hrs =0; between 5 and 10hrs =1; more than 10hrs =2)",
            "MathScore: math test score (0-100)",
            "ReadingScore: reading test score (0-100)",
            "WritingScore: writing test score (0-100)"
        ]
        for feature in features:
            st.write(feature)
    st.dataframe(df_exam_score.head())

    st.write("Without cleaning the dataset, we can plot only numerical features (as seen below).")
    num_cols = len(df_exam_score.select_dtypes(include='number').columns)
    fig, axes = plt.subplots(1, num_cols, figsize=(num_cols * 6, 6))

    # Plot histograms for all numerical columns
    for i, column in enumerate(df_exam_score.select_dtypes(include='number').columns):
        df_exam_score[column].hist(ax=axes[i], bins=10)
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Adjust layout and display
    plt.tight_layout()
    st.pyplot(fig)
    
### DATASET CLEANING ###
    st.write("# Dataset Cleaning and Handling missing values")
    df_exam_score.drop('Unnamed: 0',axis=1, inplace=True)
    
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
    
    # EthnicGroup and ParentMaritalStatus: One-hot encoding: Each ethnic group has a different binary variable.
    df_exam_score = pd.get_dummies(df_exam_score, columns=['EthnicGroup', 'ParentMaritalStatus'], dummy_na=True)

    df_exam_score.drop('EthnicGroup_nan', axis=1, inplace=True)
    df_exam_score.drop('ParentMaritalStatus_nan', axis=1, inplace=True)

    # st.dataframe(df_exam_score)
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    # df_exam_score.plot(kind='hist', bins=100, ax=axes[0], title='Score')

    # for ax in axes:
    #     ax.set_xlabel('Score')
    #     ax.set_ylabel('Frequency')

    # plt.tight_layout() # to avoid overlap
    # plt.show()
    st.write("Dataframe after cleaning.")
    st.dataframe(df_exam_score)

    st.write("### Plot exam scores")
    st.write('''The plot displays how students' scores are spread across different subjects, 
            revealing slight variations.Generally, students perform better in reading and writing, 
            as indicated by the predominance of scores above 35-40 to 100 in these subjects. 
            In the other hand, math scores show a wider range, with numerous scores falling also below 30.''')
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(df_exam_score['MathScore'], df_exam_score.index, color='blue', alpha=0.3) 
    plt.title('Math Scores')  
    plt.xlabel('Score')
    plt.ylabel('Student Index')

    plt.subplot(1, 3, 2)
    plt.scatter(df_exam_score['ReadingScore'], df_exam_score.index, color='red', alpha=0.3) 
    plt.title('Reading Scores')  
    plt.xlabel('Score')
    plt.ylabel('Student Index')

    plt.subplot(1, 3, 3)
    plt.scatter(df_exam_score['WritingScore'], df_exam_score.index, color='green', alpha=0.3) 
    plt.title('Writing Scores')  
    plt.xlabel('Score')
    plt.ylabel('Student Index')

    plt.tight_layout()
    st.pyplot(fig)

#_______________###############_______________

    st.write("decide if keep or not")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    df_exam_score['MathScore'].plot(kind='hist', bins=100, ax=axes[0], title='Math Score')
    df_exam_score['ReadingScore'].plot(kind='hist', bins=100, ax=axes[1], title='Reading Score')
    df_exam_score['WritingScore'].plot(kind='hist', bins=100, ax=axes[2], title='Writing Score')

    for ax in axes:
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')

    plt.tight_layout() # to avoid overlap
    plt.show()

    st.write("## Dataset Plot")
    st.pyplot(fig)


#_______________###############_______________


    st.write("## Correlation")
    df_exam_score.WklyStudyHours.unique()
    df_exam_score_corr = df_exam_score.corr()
    st.dataframe(df_exam_score_corr)

    st.write('''Correlation comment: the most influencing features for scores are:
                    - gender
                    - parent education 
                    - lunchtype
                    - test preparation
                    - sport practicing
                    - weekly study hours
                    - ethnic group
                We can also see that reading and writing scores are more correlated. 
                So we can say that who is good at reading is also good in writing. ''')

    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(df_exam_score_corr, annot=True, ax=ax)
    st.pyplot(fig)

    ### PLOTS GROUPBY gender###
    st.title('Correlation between Exam Scores and Gender')
    st.write("By analyzing this plot we can see that male students are better in Mathematics, instead female sudents are better in Reading and Writing.")
    
    scores_for_each_gender = df_exam_score.groupby('Gender').mean()
    scores_for_each_gender[["MathScore", "ReadingScore", "WritingScore"]]
    
    fig, ax = plt.subplots()
    scores_for_each_gender[["MathScore", "ReadingScore", "WritingScore"]].plot(kind='bar', ax=ax, rot=0) #[["MathScore", "ReadingScore", "WritingScore"]].plot(kind='bar', rot=0)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Exam Scores by Gender and Subject')
    ax.legend(title='Subject',loc='upper center', prop={'size': 5} )
    ax.set_xticklabels(['Male', 'Female'])
    st.pyplot(fig)

    ### PLOTS GROUPBY practicing sport###
    st.title('Correlation between Exam Scores and Practicing Sport')
    st.write("By this plot we can see that practicing sport regulary influence better student score, especially reading and writing scores.")

    scores_correlation_with_ethnic_group = df_exam_score.groupby('PracticeSport').mean()
    scores_correlation_with_ethnic_group[["MathScore", "ReadingScore", "WritingScore"]]

    fig, ax = plt.subplots() 
    scores_correlation_with_ethnic_group[["MathScore", "ReadingScore", "WritingScore"]].plot(kind='bar', ax=ax, rot=0) #[["MathScore", "ReadingScore", "WritingScore"]].plot(kind='bar', rot=0)
    ax.set_xlabel('Sport practicing')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Exam Scores by Sport Practicing and Subject')
    ax.legend(title='Sport Practicing', bbox_to_anchor=(1.05, 1), prop={'size': 5} )
    ax.set_xticklabels(['Never', 'Sometimes', 'Regularly'])
    st.pyplot(fig)

    ### PLOTS GROUPBY ethnic group###
    st.title('Correlation between Exam Scores and Ethnic Group')
    st.write('''By this plot we can comment:
                Students belonging to the Group A have better scores then other groups in all subjects. 
             And they are good espacially in Mathematics. Instead other groups are better in Reading and Writing. ''')

    ethnic_group_list = list(df_exam_score[['EthnicGroup_group A', 'EthnicGroup_group B', 'EthnicGroup_group C', 'EthnicGroup_group D', 'EthnicGroup_group E']])
    scores_correlation_with_ethnic_groups = df_exam_score.groupby(ethnic_group_list).mean()
    scores_correlation_with_ethnic_groups[["MathScore", "ReadingScore", "WritingScore"]]

    fig, ax =  plt.subplots()
    scores_correlation_with_ethnic_groups[["MathScore", "ReadingScore", "WritingScore"]].plot(kind='bar', ax=ax, rot=0)
    ax.set_xlabel('Ethnic Groups')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Exam Scores by Ethnic Groups and Subject')
    ax.legend(title='Subject',loc='upper right', prop={'size': 4} )
    ax.set_xticklabels(['Group A', 'Group B', 'Group C', 'Group D', 'Group E'])
    st.pyplot(fig)
#_______________########_MODEL_########_______________

    st.write("# Model")
    st.write('''The goal is to predict students' score. Initially, I explored a variety of regression algorithms, 
             including Linear Regression, Polynomial Regression, Random Forest Regression, and XGBoost Regressor. 
             After experimentation, Polynomial Regression yielded the highest accuracy.''')
    # preparation for model training and evaluation
    X = df_exam_score.drop(['MathScore','ReadingScore','WritingScore'], axis=1)
    y_math=df_exam_score['MathScore']
    y_read=df_exam_score['ReadingScore']
    y_write=df_exam_score['WritingScore']

    ## Functions
    def model_evaluation(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)

        MAE = metrics.mean_absolute_error(y_test, y_pred)
        print(f'MAE score is: {MAE}')
        MSE = metrics.mean_squared_error(y_test, y_pred)
        print(f'MSE score is: {MSE}')
        RMSE = np.sqrt(MSE)
        print(f'RMSE score is: {RMSE}')
        R2_Score = metrics.r2_score(y_test, y_pred)
        print(f'R2_Score score is: {R2_Score}')
        
        return pd.DataFrame([MAE, MSE, RMSE, R2_Score], index=['MAE', 'MSE', 'RMSE' ,'R2-Score'], columns=[model_name])


    def model_train(X,y,model):
        #split dataset by taking 80% of data for training and 20% of data for testing.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        model.fit(X_train, y_train)

        model_evaluation(model, X_test, y_test, 'Linear Reg.')
        
        return model

    # DIFFERENT MODELS TRYING AND CHOSING THAT ONE WITH BEST SCORE
    # st.write("First trial: LinearRegressor on Math Score.")
    model=LinearRegression()
    model_math=model_train(X,y_math,model)

    # st.write("Second and Third Trial: PolyRegressor of degree=2 and degree=3 on Math Score")
    # st.write("Since scores slightly drop with degree increase, degree of 2 is accepted sufficient.")

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Create and fit the polynomial regression model
    model = LinearRegression()
    model_math=model_train(X_poly,y_math,model)

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Create and fit the polynomial regression model
    model = LinearRegression()
    model_math=model_train(X_poly,y_math,model)


    # st.write("Fourth Trial: Random Forest Regression on Math Score")
    model=RandomForestRegressor(n_estimators=100)
    model_math=model_train(X,y_math,model)

    # st.write("Fifth Trial: XGBoost")
    model = XGBRegressor(learning_rate=0.05,max_depth=2,n_estimators= 150)
    model_math=model_train(X,y_read,model)

    # Create a DataFrame to display the evaluation metrics
    # Train and evaluate the math model
    model.fit(X_poly, y_math)
    y_pred_math = model.predict(X_poly)
    mse_math = mean_squared_error(y_math, y_pred_math)
    r2_math = r2_score(y_math, y_pred_math)

    # Train and evaluate the reading model
    model.fit(X_poly, y_read)
    y_pred_read = model.predict(X_poly)
    mse_read = mean_squared_error(y_read, y_pred_read)
    r2_read = r2_score(y_read, y_pred_read)

    # Train and evaluate the writing model
    model.fit(X_poly, y_write)
    y_pred_write = model.predict(X_poly)
    mse_write = mean_squared_error(y_write, y_pred_write)
    r2_write = r2_score(y_write, y_pred_write)

    # Create a DataFrame to display the evaluation metrics
    data = {
        'Model': ['Math', 'Reading', 'Writing'],
        'MSE': [mse_math, mse_read, mse_write],
        'R2 Score': [r2_math, r2_read, r2_write]
    }

    df = pd.DataFrame(data)

    st.dataframe(df)

    

if __name__ == "__main__":
    main()
