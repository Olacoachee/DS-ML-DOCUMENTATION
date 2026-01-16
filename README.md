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





## Author
Monday Olawale
