# DS-ML-DOCUMENTATION
 Insurance Claim Prediction Portfolio Building for Real-World ML Capstone
 
# Insurance Claim Prediction Using Machine Learning
# STEP 1
## 1. Project Overview
This project builds a predictive model to estimate the probability of an
insurance claim for buildings based on structural and environmental features.

## About Project
You have been appointed as the Lead Data Analyst to build a predictive model to determine if a building will have an insurance claim during a certain period or not. You will have to predict the probability of having at least one claim over the insured period of the building. The model will be based on the building characteristics. The target variable, Claim, is a:

1 if the building has at least a claim over the insured period.<br>
0 if the building doesn’t have a claim over the insured period.

## Dataset
The dataset contains building characteristics and a binary target variable
indicating claim occurrence.

## Models Used
- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Tools
- Python [Downlaod here](https://www.python.org/)
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 2. Importing Required Libraries

In this stage of the project, all necessary Python libraries required for data analysis, visualization, preprocessing, modeling, and evaluation were imported.
Core data manipulation and numerical operations were handled using Pandas and NumPy. For data visualization and exploratory analysis, Matplotlib and Seaborn were used to generate informative plots and charts.

Machine learning functionality was implemented using Scikit-learn, including tools for data splitting, feature scaling, categorical encoding, and pipeline construction to ensure consistent preprocessing. Multiple classification models—Logistic Regression, Decision Tree, and Random Forest—were imported to enable model comparison and selection.

Evaluation metrics such as accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report were also included to allow for comprehensive assessment of model performance. Finally, warnings were suppressed to improve notebook readability and focus on key outputs.

This setup ensured a structured, reproducible, and professional workflow for the end-to-end machine learning pipeline.

``` 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings("ignore")
```
#  STEP 2: Data Understanding and Initial Inspection
### 1. View the Data
At this stage, the dataset was loaded from an Excel file into a Pandas DataFrame to begin the data understanding process. The read_excel() function was used to import the training dataset, ensuring that all features and target variables were correctly captured.

The first few rows of the dataset were displayed using df.head() to gain an initial overview of the data structure, including column names, data types, and sample observations. This preliminary inspection helped verify that the data was loaded successfully and provided early insights into the variables available for analysis and modeling.
This step laid the foundation for subsequent data cleaning, exploratory data analysis, and feature engineering.

```
df = pd.read_excel(
    r"C:\Users\User\Desktop\DATA SCIENCE & ML\PROJECT CAPSTONE\Train_data.xlsx"
)

df.head()
```
<img width="721" height="120" alt="Data View" src="https://github.com/user-attachments/assets/b994a002-d7fb-4773-8f21-d3352615d5c2" /> <br> Table 1: Data View

## 2. Basic Information

The df.info() method was used to obtain a concise summary of the dataset, including the total number of observations, number of features, data types of each column, and the presence of missing values.

This step provided clarity on:The overall size of the dataset, which variables are numerical or categorical, columns containing missing values that may require preprocessing,
Memory usage, and data consistency. Understanding the dataset structure at this stage was essential for guiding data cleaning, feature engineering, and selecting appropriate preprocessing techniques before model building.

```
df.info()
```
<img width="255" height="247" alt="Basic Information" src="https://github.com/user-attachments/assets/3f7a47ef-8ea6-4d1d-a706-21398753fe1f" /> <br> Figure 1: Data Basic Information

## 3. Basic Dataset Information
The df.info() function was used to examine the structural properties of the dataset. This includes the number of rows and columns, data types of each feature, non-null counts, and overall memory usage. This initial inspection is critical for designing an effective data preprocessing strategy and ensuring the dataset is suitable for exploratory data analysis and machine learning modeling.

```
df.describe()
```
<img width="503" height="213" alt="Data Summary" src="https://github.com/user-attachments/assets/77c53c52-3a2b-4f08-b6f2-970eda15b42f" /> <br> Table 2: Statistical summary

## 4. Missing Values Analysis
The df.isnull().sum() function was used to identify and quantify missing values across all features in the dataset. Sorting the results in descending order helped highlight variables with the highest number of missing entries. Understanding the pattern and scale of missing values is a crucial step in data preprocessing, as improper handling of missing data can significantly impact model performance and reliability.

```
df.isnull().sum().sort_values(ascending=False)
```
<img width="144" height="178" alt="Missing Value" src="https://github.com/user-attachments/assets/279b6a4b-b9c4-47ea-a7f4-cad251a25da5" /> <br> Figure 2: Missing Values

# STEP 3: Target Variable Check
## 1. Target Distribution
Before cleaning, it is always good to check the target distribution:
The df['Claim'].value_counts(normalize=True) was used to check whether the dataset is imbalanced and 
Whether special techniques (e.g. class weights) may be needed.

```
df['Claim'].value_counts(normalize=True)
```

## 2. Distribution of Insurrance Claim Plot
A count plot was created to visualize the distribution of the target variable, Claim, which indicates whether a building had at least one insurance claim during the insured period. Understanding the distribution of the target variable is important because class imbalance can influence model performance and evaluation metrics. This insight helps inform decisions on model selection, evaluation strategy, and the potential need for techniques such as class weighting or resampling.

```
sns.countplot(x='Claim', data=df)
plt.title("Distribution of Insurance Claims")
plt.show()
```
<img width="388" height="266" alt="Distribution of Ins Claim" src="https://github.com/user-attachments/assets/9f231a80-b5dd-4818-9ffa-fcabf47eb969" /> <br>
Figure 3: Distribution of Insurrance Claim 

# STEP 4: Data Cleaning & Preprocessing

This is one of the most important sections of this project. As the quality of input data directly affects the performance and reliability of machine learning models.

To preserve the integrity of the original dataset, a copy of the raw data was created using:

```
df_clean = df.copy()
```
This ensures that:

- The original dataset remains unchanged and can be referenced at any time

- All cleaning, transformations, and preprocessing steps are applied safely on a separate dataset

- Errors or experimental changes can be reversed without data loss

Working on a clean copy allows for a structured, transparent, and reproducible preprocessing workflow, which is essential for a well-documented data science project.

# 3 Fix NumberOfWindows Column
The NumberOfWindows column contained invalid entries represented by ".", which prevented the column from being treated as a numerical variable. Since this feature is expected to be numeric, corrective preprocessing was required. Median imputation was chosen because it is robust to outliers and is well-suited for skewed numerical distributions.

```
df_clean['NumberOfWindows'] = (
    pd.to_numeric(
        df_clean['NumberOfWindows'].astype(str).str.strip(),
        errors='coerce'
    )
)

df_clean['NumberOfWindows'].fillna(
    df_clean['NumberOfWindows'].median(),
    inplace=True
)
df_clean['NumberOfWindows'].isna().sum()
df_clean['NumberOfWindows'].dtype
```
# 4 Handle Building Dimension Missing Values
The Building Dimension feature is a numerical variable that contained missing values. Since this variable represents a continuous measurement, it is important to handle missing data in a way that preserves the original distribution. Median imputation was applied to replace missing values, this was chosen instead of the mean to reduce the influence of potential outliers. After imputation, the column no longer contains missing values and is now suitable for exploratory analysis and machine learning modeling.

```
df_clean['Building Dimension'].fillna(
    df_clean['Building Dimension'].median(), inplace=True
)
df_clean['Building Dimension'].isna().sum()
```
# 5 Handle Date_of_Occupancy
The Date_of_Occupancy column is a date-based feature with missing values. Since machine learning models cannot directly interpret raw date formats, feature engineering was required to extract meaningful numerical information.
The following steps were applied:

- Converted the column to a datetime format, coercing invalid entries into NaN

- Created a new feature, Building_Age, by subtracting the year of occupancy from the YearOfObservation

- Imputed missing values in the newly created Building_Age feature using the median

- Removed the original Date_of_Occupancy column to avoid redundancy

This transformation improves model interpretability by capturing the age of the building, which is a more informative and usable predictor for insurance claim risk than the raw date itself.
```
df_clean['Date_of_Occupancy'] = pd.to_datetime(
    df_clean['Date_of_Occupancy'], errors='coerce'
)

# Create Building Age feature
df_clean['Building_Age'] = (
    df_clean['YearOfObservation'] - df_clean['Date_of_Occupancy'].dt.year
)

# Fill missing ages with median
df_clean['Building_Age'].fillna(
    df_clean['Building_Age'].median(), inplace=True
)

# Drop original date column
df_clean.drop(columns=['Date_of_Occupancy'], inplace=True)
```

# 6 Drop Irrelevant Columns
Certain columns in the dataset do not contribute to predicting insurance claims and may introduce noise or data leakage if retained. In particular, the Customer Id column serves only as a unique identifier and contains no predictive information. This step helps create a cleaner and more reliable feature set for machine learning.

```
df_clean.columns = df_clean.columns.str.strip()
df_clean.drop(columns=['Customer Id'], inplace=True)
```
# 7 Garden (Categorical → use mode)
The Garden feature is a categorical variable with missing values. Since categorical data represents discrete categories rather than continuous measurements, mode imputation is an appropriate strategy. After this step, the Garden feature contains no missing values and is ready for encoding and modeling.

```
df_clean['Garden'].fillna(
    df_clean['Garden'].mode()[0],
    inplace=True
)
```

# 8 Geo_Code Identifier / Location code
The Geo_Code feature represents a geographical or location-based identifier and is categorical in nature. Although it is encoded numerically, it does not carry ordinal or continuous meaning. After imputation, the Geo_Code column contains no missing values and is suitable for categorical encoding during the modeling phase.

```
df_clean['Geo_Code'].fillna(
    df_clean['Geo_Code'].mode()[0],
    inplace=True
)
```



# 9 Check Data Types After Cleaning
At this stage of the preprocessing pipeline, a final validation was performed to ensure the dataset is fully prepared for machine learning modeling.

Using df_clean.info() and df_clean.isna().sum(), the following conditions were confirmed:

- All numerical features are correctly stored in numeric data types

- All categorical features are stored as object types

- There are no remaining missing values in any column

- The dataset structure is consistent and free from data quality issues

This final check is crucial because machine learning algorithms require clean, complete, and properly formatted data. Ensuring these conditions are met helps prevent runtime errors, improves model stability, and enhances predictive performance. With the dataset now fully cleaned and validated, it is ready for feature encoding, scaling, and model training.

```
df_clean.info()
```

```
df_clean.isna().sum()
```

# STEP 5: EXPLORATORY DATA ANALYSIS (EDA)
The distribution of the target variable, Claim, was analyzed to understand the proportion of buildings that experienced at least one insurance claim during the insured period versus those that did not.
```
df_clean['Claim'].value_counts(normalize=True)
```
# 1 Target variable plotting
A count plot was used to visually represent the distribution of the target variable, Claim, where:

- 0 indicates buildings without insurance claims

- 1 indicates buildings with at least one insurance claim
```
sns.countplot(x='Claim', data=df_clean)
plt.title("Distribution of Insurance Claims")
plt.xlabel("Claim (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
```

<img width="388" height="266" alt="Distribution of Ins Claim" src="https://github.com/user-attachments/assets/d728fbac-99a2-491f-80c2-ad98e3164a63" /> <br>
Figure 4: Distribution of the Target Variable (Claim vs No Claim)

The distribution of the target variable shows that non-claim cases are more
frequent than claim cases. This indicates a class imbalance, which is common
in insurance datasets and must be considered during model evaluation.

# 2 Numerical Features vs Claim
In this step, I identified all the numerical columns in the cleaned dataset (df_clean) by selecting columns with data types int64 and float64. This helps in understanding which variables can be analyzed quantitatively against the target variable Claim for further statistical analysis or visualization.

```
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
num_cols
```
# 2.1 Distribution Plotting
In this step, I visualized the distribution of numerical features against the target variable Claim. For each numerical column (excluding Claim itself), a boxplot was generated to show how the values vary between different claim outcomes. This helps identify potential patterns, trends, or outliers in the data that may influence claim occurrences.
```
for col in num_cols:
    if col != 'Claim':
        sns.boxplot(x='Claim', y=col, data=df_clean)
        plt.title(f"{col} vs Claim")
        plt.show()
```

<img width="387" height="265" alt="YearOfObservation vs Claim" src="https://github.com/user-attachments/assets/5202b44a-45e5-46c5-a8ca-4d9b548094af" /> <br> Figure 5: Year of Observation vs Claim

<img width="367" height="278" alt="Insured_Period vs Claim" src="https://github.com/user-attachments/assets/406b9cc6-55b4-401b-b207-16ae233475d4" /> <br> Figure 6: Insured_Period vs Claim

<img width="372" height="272" alt="Residential vs Claim" src="https://github.com/user-attachments/assets/847722ee-f8bd-4a1e-a430-a8e63af05672" /> <br> Figure 7: Residential vs Claim

<img width="384" height="278" alt="Building Dimension vs Claim" src="https://github.com/user-attachments/assets/37c12340-9288-4116-a511-3aaca014c366" /> <br> Figure 8: Building Dimension vs Claim

<img width="358" height="268" alt="Building_Type vs Claim" src="https://github.com/user-attachments/assets/bf6d19a3-5925-486d-b4ca-dd06b68de312" /> <br> Figure 9: Building_Type vs Claim

<img width="363" height="267" alt="NumberOfWindows vs Claim" src="https://github.com/user-attachments/assets/2efd27da-481a-40b2-8af9-52dd42dc8646" /> <br> Figure 10: NumberOfWindows vs Claim

<img width="378" height="269" alt="Building_Age vs Claim" src="https://github.com/user-attachments/assets/491aafe1-877d-4711-ad18-051529177a6a" /> <br> Figure 11: Building_Age vs Claim

Both claim groups exhibit an identical distribution of YearOfObservation values. This suggests that the observation year is not a determining factor for whether a claim is made in this dataset. <br>
The distribution of the Insured_Period variable is visually the same whether a claim was made (1) or not (0). This suggests that the duration of the insurance period is not a distinguishing factor for claim status in this dataset. <br>
All properties in the dataset are categorized as residential, regardless of their claim status (0 or 1). This means that the residential status is constant across all observations and does not differentiate between those who made a claim and those who did not. <br>
The median Building Dimension is notably higher for properties that had a claim (labeled 1) compared to those without a claim (labeled 0). This suggests that larger building dimensions might be associated with a higher likelihood of a claim in this dataset. <br>
The distribution of the Building_Type variable is visually the same whether a claim was made (1) or not (0). This suggests that the specific building type number is not a distinguishing factor for claim status in this dataset. <br>
The median number of windows is higher (five) for properties with a claim compared to those without a claim (four). This suggests that having more windows might slightly increase the likelihood of a claim in this dataset. <br>
The distribution of Building_Age is visually identical for properties with a claim (1) and those without a claim (0). This suggests that building age is not a distinguishing factor for claim status in this dataset.

# 3 Categorical Features vs Claim
### Identify categorical columns
In this step, I identified all categorical columns in the cleaned dataset (df_clean) by selecting columns with data type object. These columns can be analyzed against the target variable Claim to explore relationships, patterns, or trends in categorical data.

```
cat_cols = df_clean.select_dtypes(include='object').columns
cat_cols
```
# 3.1 Distribution Plotting
In this step, I analyzed the relationship between categorical features and the target variable Claim. For each categorical column, I calculated the claim rate (average of Claim) per category and visualized it as a bar chart. This approach highlights which categories are associated with higher or lower probabilities of a claim, providing insights for feature importance and further modeling.
```
for col in cat_cols:
    claim_rate = (
        df_clean.groupby(col)['Claim']
        .mean()
        .sort_values(ascending=False)
    )
    
    claim_rate.plot(kind='bar')
    plt.title(f"Claim Rate by {col}")
    plt.ylabel("Claim Probability")
    plt.show()
```













## Author
Monday Olawale
