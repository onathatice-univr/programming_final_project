# STUDENTS EXAM SCORE ANALYSIS
The project aims to analyze a dataset containing lifestyle and exam score information about students. Its primary objective is to predict students' scores using regression algorithms. After experimenting with various algorithms, polynomial regression emerged as the most effective method.

## Files Included:
• VR503361_final_project.ipynb: Initial Python code to fulfill project goals.
• function.py: Contains functions utilized by main script files.
• script_with_streamlit.py: Python code with Streamlit for improved project presentation.
• exam_score_dataset.csv: Dataset used for the project.
• requirements.txt: Lists necessary Python libraries.

## [Dataset Information](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores?select=Expanded_data_with_more_features.csv)
    • Gender: Gender of the student (male/female)
    • EthnicGroup: Ethnic group of the student (group A to E)
    • ParentEduc: Parent(s) education background (from some_highschool to master's degree)
    • LunchType: School lunch type (standard or free/reduced)
    • TestPrep: Test preparation course followed (completed or none)
    • ParentMaritalStatus: Parent(s) marital status (married/single/widowed/divorced)
    • PracticeSport: How often the student parctice sport (never/sometimes/regularly)
    • IsFirstChild: If the child is first child in the family or not (yes/no)
    • NrSiblings: Number of siblings the student has (0 to 7)
    • TransportMeans: Means of transport to school (schoolbus/private)
    • WklyStudyHours: Weekly self-study hours(less that 5hrs; between 5 and 10hrs; more than 10hrs)
    • MathScore: math test score(0-100)
    • ReadingScore: reading test score(0-100)
    • WritingScore: writing test score(0-100)

# MACHINE LEARNING MODEL:
The goal is to predict students' score. Initially, I explored a variety of regression algorithms, including Linear Regression, Polynomial Regression, Random Forest Regression, and XGBoost Regressor. After experimentation, Polynomial Regression yielded the highest accuracy.
