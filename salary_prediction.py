import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # Ignore warnings for cleaner output

# --- Global Variables ---
# These variables store data and model components, accessible throughout the program.
df = None # Main DataFrame for salary data
X = None  # Features for the model
y = None  # Target variable (Salary)
categorical_cols = None # List of categorical column names in X
numerical_cols = None   # List of numerical column names in X
preprocessor = None     # Preprocessing pipeline (scales numerical, one-hot encodes categorical)
model = None            # Trained Linear Regression model
processed_feature_names = None # Feature names after preprocessing
onehot_features = None  # Names of features created by one-hot encoding

# --- 1. Initial Model Setup ---
# Sets up the baseline model by loading data, preprocessing, training, and evaluating.
def initial_model_setup():
    global df, X, y, categorical_cols, numerical_cols, preprocessor, model, processed_feature_names, onehot_features

    print("\n--- Initial Model Setup ---")
    df = pd.read_csv('/content/salary_prediction_data.csv')
    print("Dataset loaded successfully.")

    X = df.drop('Salary', axis=1) # Features are all columns except 'Salary'
    y = df['Salary'] # Target variable is 'Salary'

    categorical_cols = X.select_dtypes(include=['object']).columns # Columns with object data type
    numerical_cols = X.select_dtypes(include=np.number).columns     # Columns with numerical data type

    # Defines preprocessing steps: scale numerical, one-hot encode categorical.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X) # Apply preprocessing to features

    # Get new feature names after one-hot encoding
    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    processed_feature_names = list(numerical_cols) + list(onehot_features)

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

    # Split data into 80% training and 20% testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

    model = LinearRegression() # Initialize a Linear Regression model
    model.fit(X_train, y_train) # Train the model
    print("Baseline model trained successfully.")

    y_pred = model.predict(X_test) # Make predictions on the test set

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\nInitial Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

# --- 2. User Input and Data Management ---
# Collects new employee details from the user with validation.
def get_user_input_data():
    print("\nEnter details for the new salary record:")
    # Predefined valid options for categorical inputs
    education_options = ['High School', 'Bachelor', 'Master', 'PhD']
    location_options = ['Urban', 'Suburban', 'Rural']
    job_title_options = ['Manager', 'Director', 'Analyst', 'Engineer']
    gender_options = ['Male', 'Female']

    # Input loops with validation for each feature
    while True:
        education = input(f"Education ({', '.join(education_options)}): ").strip()
        if education in education_options:
            break
        else:
            print("Invalid education. Please choose from the given options.")

    while True:
        try:
            experience = int(input("Experience (years): ").strip())
            if experience >= 0:
                break
            else:
                print("Experience cannot be negative.")
        except ValueError:
            print("Invalid input. Please enter a number for experience.")

    while True:
        location = input(f"Location ({', '.join(location_options)}): ").strip()
        if location in location_options:
            break
        else:
            print("Invalid location. Please choose from the given options.")

    while True:
        job_title = input(f"Job Title ({', '.join(job_title_options)}): ").strip()
        if job_title in job_title_options:
            break
        else:
            print("Invalid job title. Please choose from the given options.")

    while True:
        try:
            age = int(input("Age: ").strip())
            if 0 < age < 120:
                break
            else:
                print("Invalid age. Please enter a reasonable age.")
        except ValueError:
            print("Invalid input. Please enter a number for age.")

    while True:
        gender = input(f"Gender ({', '.join(gender_options)}): ").strip()
        if gender in gender_options:
            break
        else:
            print("Invalid gender. Please choose from the given options.")

    # Create a pandas Series from user input (Salary is not asked here)
    new_data = pd.Series({
        'Education': education,
        'Experience': experience,
        'Location': location,
        'Job_Title': job_title,
        'Age': age,
        'Gender': gender
    })
    return new_data

# Adds new data (employee record) to the main DataFrame.
def add_data_to_df(new_data_series, salary=np.nan):
    global df

    new_record_df = pd.DataFrame([new_data_series]) # Convert Series to DataFrame row

    # Add NaN as a placeholder for Salary if not provided, to match DataFrame structure
    if 'Salary' not in new_record_df.columns:
        new_record_df['Salary'] = salary

    # Ensure new record has all original columns, filling missing with NaN, then reorder
    for col in df.columns:
        if col not in new_record_df.columns:
            new_record_df[col] = np.nan
    new_record_df = new_record_df[df.columns]

    df = pd.concat([df, new_record_df], ignore_index=True) # Append new record
    print("New data added successfully to the dataset.")
    print(f"Current dataset size: {len(df)} records.")

# Displays a specific data row from the DataFrame based on user-provided index.
def display_specific_data():
    global df
    while True:
        try:
            index_to_display = int(input(f"Enter the row index to display (0 to {len(df) - 1}): ").strip())
            if 0 <= index_to_display < len(df):
                print(f"\nData for index {index_to_display}:")
                print(df.iloc[index_to_display])
                break
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(df) - 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer for the index.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Compares details of two employees by their indices.
def compare_two_employees():
    global df
    print("\n--- Compare Two Employees ---")
    # Input and validate index for the first employee
    while True:
        try:
            idx1 = int(input(f"Enter the index of the first employee (0 to {len(df) - 1}): ").strip())
            if 0 <= idx1 < len(df):
                break
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(df) - 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer for the index.")

    # Input and validate index for the second employee
    while True:
        try:
            idx2 = int(input(f"Enter the index of the second employee (0 to {len(df) - 1}): ").strip())
            if 0 <= idx2 < len(df):
                break
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(df) - 1}.")
        except ValueError:
            print("Invalid input. Please enter an integer for the index.")

    print(f"\n--- Comparison of Employee {idx1} and {idx2} ---")
    # Display selected employee details side-by-side
    comparison_df = df.iloc[[idx1, idx2]].transpose()
    comparison_df.columns = [f'Employee {idx1}', f'Employee {idx2}']
    display(comparison_df)

# --- 3. Prediction and Retraining ---
# Predicts salary for a new data entry using the current trained model.
def predict_with_current_model(new_data_series):
    global preprocessor, model, X, processed_feature_names

    if model is None or preprocessor is None:
        print("Error: Model or preprocessor not initialized. Please set up/retrain the model first.")
        return

    new_data_df = pd.DataFrame([new_data_series]) # Convert new data to DataFrame

    # Align new data columns with original training features, adding NaN for missing ones
    original_X_cols = X.columns.drop('Salary', errors='ignore') if 'Salary' in X.columns else X.columns
    missing_cols_in_new_data = set(original_X_cols) - set(new_data_df.columns)
    for c in missing_cols_in_new_data:
        new_data_df[c] = np.nan
    new_data_df = new_data_df[original_X_cols]

    new_data_processed = preprocessor.transform(new_data_df) # Preprocess new data

    # Convert processed data to DataFrame with correct feature names
    new_data_processed_df = pd.DataFrame(new_data_processed, columns=processed_feature_names)

    predicted_salary = model.predict(new_data_processed_df)[0] # Make prediction
    print(f"Predicted Salary: ₹{predicted_salary:,.2f}") # Display salary in Indian Rupees
    return predicted_salary

# Retrains the model using all available data, including newly added entries.
def retrain_model():
    global df, X, y, categorical_cols, numerical_cols, preprocessor, model, processed_feature_names, onehot_features

    print("\nRetraining model...")

    current_df = df.copy() # Work on a copy to avoid altering original df during cleaning

    # Drop rows without a 'Salary' value, as they cannot be used for training
    current_df.dropna(subset=['Salary'], inplace=True)

    if current_df.empty:
        print("Cannot retrain: No complete salary records available after removing entries without salary.")
        return

    X = current_df.drop('Salary', axis=1) # Update features
    y = current_df['Salary'] # Update target

    categorical_cols = X.select_dtypes(include=['object']).columns # Re-identify categorical columns
    numerical_cols = X.select_dtypes(include=np.number).columns   # Re-identify numerical columns

    # Re-initialize preprocessor to fit any new categories from added data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    X_processed = preprocessor.fit_transform(X) # Re-fit and transform features

    # Update feature names after re-fitting preprocessor
    onehot_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    processed_feature_names = list(numerical_cols) + list(onehot_features)

    X_processed_df = pd.DataFrame(X_processed, columns=processed_feature_names)

    # Re-split data into training and testing sets
    if len(X_processed_df) < 2: # Check for minimum samples for splitting
        print("Not enough data to split and retrain the model after cleaning.")
        return
    elif len(X_processed_df) == 2:
        # If only 2 samples, train with all data (no test set for evaluation)
        X_train, y_train = X_processed_df, y
        X_test, y_test = pd.DataFrame(), pd.Series()
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

    model = LinearRegression() # Re-initialize model
    model.fit(X_train, y_train) # Re-train model

    # Evaluate and print new performance metrics if a test set exists
    if not X_test.empty:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Retrained Successfully!")
        print(f"New Mean Absolute Error (MAE): {mae:,.2f}")
        print(f"New Root Mean Squared Error (RMSE): {rmse:,.2f}")
        print(f"New R-squared (R2) Score: {r2:.4f}")
    else:
        print("Model Retrained Successfully! (No test set for evaluation due to limited data)")

# --- 4. Data Visualization Functions ---
# Performs various visualizations to explore the dataset.
def perform_visualizations():
    global df
    if df is None or df.empty:
        print("Please load or add data before performing visualizations.")
        return

    print("\n--- Performing Data Visualizations ---")

    # Visualize Distributions of Numerical Features
    print("\nVisualizing Distributions of Numerical Features...")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df['Experience'], kde=True)
    plt.title('Distribution of Experience')
    plt.subplot(1, 3, 2)
    sns.histplot(df['Age'], kde=True)
    plt.title('Distribution of Age')
    plt.subplot(1, 3, 3)
    sns.histplot(df['Salary'], kde=True)
    plt.title('Distribution of Salary')
    plt.tight_layout()
    plt.show()

    # Visualize Distributions of Categorical Features
    print("\nVisualizing Distributions of Categorical Features...")
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1)
    sns.countplot(y=df['Education'], order=df['Education'].value_counts().index)
    plt.title('Distribution of Education Levels')
    plt.ylabel('')
    plt.subplot(1, 4, 2)
    sns.countplot(y=df['Location'], order=df['Location'].value_counts().index)
    plt.title('Distribution of Locations')
    plt.ylabel('')
    plt.subplot(1, 4, 3)
    sns.countplot(y=df['Job_Title'], order=df['Job_Title'].value_counts().index)
    plt.title('Distribution of Job Titles')
    plt.ylabel('')
    plt.subplot(1, 4, 4)
    sns.countplot(y=df['Gender'], order=df['Gender'].value_counts().index)
    plt.title('Distribution of Gender')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # Visualize Relationship Between Numerical Features and Salary
    print("\nVisualizing Relationship Between Numerical Features and Salary...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Experience', y='Salary', data=df)
    plt.title('Experience vs. Salary')
    plt.xlabel('Experience (Years)')
    plt.ylabel('Salary (₹)')
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Age', y='Salary', data=df)
    plt.title('Age vs. Salary')
    plt.xlabel('Age')
    plt.ylabel('Salary (₹)')
    plt.tight_layout()
    plt.show()

    # Visualize Relationship Between Categorical Features and Salary
    print("\nVisualizing Relationship Between Categorical Features and Salary...")
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 4, 1)
    sns.boxplot(x='Education', y='Salary', data=df, order=df['Education'].value_counts().index)
    plt.title('Education vs. Salary')
    plt.xlabel('')
    plt.ylabel('Salary (₹)')
    plt.xticks(rotation=45, ha='right')
    plt.subplot(1, 4, 2)
    sns.boxplot(x='Location', y='Salary', data=df, order=df['Location'].value_counts().index)
    plt.title('Location vs. Salary')
    plt.xlabel('')
    plt.ylabel('Salary (₹)')
    plt.xticks(rotation=45, ha='right')
    plt.subplot(1, 4, 3)
    sns.boxplot(x='Job_Title', y='Salary', data=df, order=df['Job_Title'].value_counts().index)
    plt.title('Job Title vs. Salary')
    plt.xlabel('')
    plt.ylabel('Salary (₹)')
    plt.xticks(rotation=45, ha='right')
    plt.subplot(1, 4, 4)
    sns.boxplot(x='Gender', y='Salary', data=df, order=df['Gender'].value_counts().index)
    plt.title('Gender vs. Salary')
    plt.xlabel('')
    plt.ylabel('Salary (₹)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Visualize Correlation Matrix of Numerical Features
    print("\nVisualizing Correlation Matrix of Numerical Features...")
    correlation_df_numerical = df[['Experience', 'Age', 'Salary']]
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df_numerical.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
    print("Visualizations completed.")

# --- 5. Interactive Menu Loop ---
# Displays the menu and handles user choices for interacting with the system.
def main_interactive_loop():
    print("\n--- Salary Prediction Interactive System ---")
    initial_model_setup() # Run initial model setup when the system starts

    while True:
        print("\nChoose an option:")
        print("1: Add New Salary Data")
        print("2: Predict Salary for New Entry")
        print("3: Retrain Model")
        print("4: Display Specific Data Entry")
        print("5: Compare Two Employees")
        print("6: Perform Data Visualizations")
        print("7: Exit")

        choice = input("Enter your choice (1-7): ").strip()

        if choice == '1':
            new_data_series = get_user_input_data()
            add_data_to_df(new_data_series)
        elif choice == '2':
            new_data_series = get_user_input_data()
            predict_with_current_model(new_data_series)
        elif choice == '3':
            retrain_model()
        elif choice == '4':
            display_specific_data()
        elif choice == '5':
            compare_two_employees()
        elif choice == '6':
            perform_visualizations() # Call the visualization function
        elif choice == '7':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")

# --- Start System ---
# Call the main interactive loop to begin the program.
main_interactive_loop()
