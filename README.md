# Salary Prediction Using Machine Learning Regression

## My Name :
Uttam Kumar Tripathi

## Project Description
This project predicts employee salaries using machine learning regression techniques. 
The system analyzes multiple employee attributes such as education level, experience, 
job title, location, age, and gender to estimate the expected salary.

The project also includes an interactive system where users can add new employee data, 
predict salary, retrain the machine learning model, and visualize the dataset using graphs.

This project demonstrates how machine learning can be applied to real-world problems 
such as salary estimation and data-driven decision making.

---
# Dataset Information

The dataset used in this project contains employee information that influences salary.

## Features in Dataset

Education – Education level of the employee (High School, Bachelor, Master, PhD)

Experience – Number of years of work experience

Location – Work location (Urban, Suburban, Rural)

Job Title – Position of the employee (Manager, Director, Analyst, Engineer)

Age – Age of the employee

Gender – Gender of the employee

Salary – Target variable that represents the employee's salary

The machine learning model learns patterns from these features to predict salary.

---
# Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

---
# Machine Learning Methodology

The following steps are used to build the salary prediction system:

1. **Data Loading**
   
   The dataset is loaded using Pandas.

2. **Data Preprocessing**
   
   - Numerical features are scaled using StandardScaler
   - Categorical features are converted using OneHotEncoder
   - ColumnTransformer is used to combine preprocessing steps

3. **Train-Test Split**
   
   The dataset is divided into:
   
   - 80% Training Data
   - 20% Testing Data

4. **Model Training**
   
   A Linear Regression model is trained using Scikit-learn.

5. **Model Evaluation**
   
   The performance of the model is evaluated using:

   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score

6. **Prediction**

   The trained model predicts the salary of new employees based on their input features.

---
# Features of the System

This project includes several functionalities:

Add new employee data to the dataset

Predict salary for a new employee

Retrain the machine learning model with updated data

Display specific employee records

Compare two employees

Perform data visualization

---
# Data Visualization

The system generates multiple visualizations to understand the dataset:

Distribution of Experience, Age, and Salary

Distribution of categorical features such as Education, Location, Job Title, and Gender

Scatter plots showing relationships between:

Experience and Salary

Age and Salary

Box plots comparing salary across different categories

Correlation heatmap of numerical features

These visualizations help analyze how different factors affect salary.

---
# Example Output

Predicted Salary for 5 years experience = 55000  
Predicted Salary for 10 years experience = 92000

The system predicts the salary based on the given employee information.

---
# Conclusion

This project demonstrates how machine learning regression models can be used 
to predict employee salaries using multiple features.

It combines data preprocessing, machine learning, visualization, and an 
interactive system to create a complete data science application.
