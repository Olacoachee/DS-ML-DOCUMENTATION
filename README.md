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
<img width="361" height="269" alt="ClaimRate by BuildingPainted" src="https://github.com/user-attachments/assets/f518924f-70f3-40da-b742-255c9839b7ff" /> <br> Figure 12: ClaimRate by BuildingPainted

<img width="363" height="275" alt="ClaimRate by BuildingFenced" src="https://github.com/user-attachments/assets/329ed30f-632e-414c-8b47-b860271b96de" /> <br> Figure 13: ClaimRate by BuildingFenced

<img width="354" height="269" alt="ClaimRate by Garden" src="https://github.com/user-attachments/assets/55e69df5-972a-4207-a91d-4f38393dc747" /> <br> Figure 14: ClaimRate by Garden

<img width="370" height="276" alt="ClaimRate by Settlement" src="https://github.com/user-attachments/assets/baf19637-a783-478d-9a34-df47e9886c63" /> <br> Figure 15: ClaimRate by Settlement

<img width="355" height="295" alt="ClaimRate by GeoCode" src="https://github.com/user-attachments/assets/56e0c438-421d-4dcf-a9ad-2c098e6a6ae1" /> <br> Figure 16: ClaimRate by GeoCode

The claim probability is higher for buildings that are painted (category 'Y', approx. 0.23) compared to those that are not painted (category 'N', approx. 0.205). This indicates that painted buildings have a slightly higher likelihood of having a claim in this dataset. <br>
Buildings that are not fenced (category 'N') have a higher claim probability (approx. 0.25) than those that are fenced (category 'Y', approx. 0.205). This indicates that fenced buildings have a lower likelihood of having a claim in this dataset. <br.
Buildings categorized as 'O' have a higher claim probability (approx. 0.25) than those categorized as 'Y' (approx. 0.20). This suggests that properties with garden status 'Y' have a lower likelihood of having a claim in this dataset. <br>
Settlements labeled as 'R' have a notably higher claim probability (0.25) compared to those labeled as 'U' (approx. 0.21). This indicates that a property's settlement type is a potential differentiator for claim status in this dataset. <br>
The claim probability varies drastically depending on the Geo_Code, ranging from 100% for some areas down to nearly 0% for others. This indicates that geographic location is a strong predictor of claim status within this dataset.

# 4 Correlation Analysis
### Correlation matrix plotting
In this step, I examined the relationships between numerical features by computing a correlation matrix. The correlations were visualized using a heatmap, with annotated values to show the strength and direction of relationships between variables. This helps identify highly correlated features, potential multicollinearity, and variables that may have stronger associations with the target variable.
```
plt.figure(figsize=(10,6))
sns.heatmap(
    df_clean[num_cols].corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()
```

<img width="456" height="324" alt="Correlation Matrix" src="https://github.com/user-attachments/assets/840be94c-6bc5-4b71-9ea0-38f756ebf0a1" /> <br> Figure 17: Correlation Matrix

The strongest relationship in the data is a high positive correlation (0.96) between YearOfObservation and Building_Age.
The variable Claim has weak positive correlations with Building Dimension (0.30), NumberOfWindows (0.17), and Insured_Period (0.09). <br>
The correlation matrix shows moderate relationships between building-related variables and claim occurrence. No extreme multicollinearity is observed, making the dataset suitable for machine learning models.

# STEP 6: MODELING PREPROCESSING
## Proceed to Modeling Preprocessing will include:
### 1. Separate Features and Target
In this step, the dataset was prepared for modeling by separating the independent variables (features) from the dependent variable (target). All predictor variables were assigned to X, while the target variable Claim was isolated as y, forming the basis for subsequent preprocessing and model training.
```
X = df_clean.drop(columns='Claim')
y = df_clean['Claim']
```
### 2. Identify Numerical and Categorical Features
In this step, the feature set (X) was further organized by identifying numerical and categorical variables based on their data types. Numerical features were selected as columns with int64 and float64 types, while categorical features were identified as columns with object type. This separation enables appropriate preprocessing techniques, such as scaling for numerical variables and encoding for categorical variables, in subsequent modeling steps.

```
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include='object').columns

numeric_features, categorical_features
```
### 3. CONVERT ALL CATEGORICAL COLUMNS TO STRING
In this step, all identified categorical features were explicitly converted to the string data type. This ensures consistency in data types and prevents potential issues during categorical encoding in the modeling pipeline, particularly when applying techniques such as one-hot encoding.
```
for col in categorical_features:
    df_clean[col] = df_clean[col].astype(str)
```
### 4. Train–Test Split

#### I split before scaling/encoding to avoid data leakage
In this step, the dataset was split into training and testing sets to enable unbiased model evaluation. The split was performed before scaling and encoding to prevent data leakage. An 80/20 split was used, with stratification applied to the target variable (Claim) to preserve class distribution across both sets, and a fixed random state to ensure reproducibility.
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## 5 Building the Preprocessing Pipelines
### Numerical Pipeline, Median imputation, Standard scaling, Categorical Pipeline, Most frequent imputation, and One-Hot Encoding
In this step, separate preprocessing pipelines were created for numerical and categorical features. Numerical variables were processed using median imputation to handle missing values and standard scaling to normalize feature ranges. Categorical variables were handled with most-frequent imputation for missing values, followed by one-hot encoding to convert categories into a machine-learning-friendly format while safely ignoring unseen categories.

```
from sklearn.impute import SimpleImputer
numeric_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```
### 6 Combine Pipelines with ColumnTransformer
In this step, the numerical and categorical preprocessing pipelines were combined using a ColumnTransformer. This allows each feature type to be processed with the appropriate transformations in a single, unified preprocessing step. The resulting preprocessor ensures consistent and reproducible data preparation before model training.
```
preprocessor = ColumnTransformer(
    transformers=[
('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
]
)
```

# STEP 7: MODEL BUILDING
In this aspect, I will train 3 different models, including Logistic Regression (baseline), Decision Tree, and Random Forest. However, all the models will use the same preprocessing pipeline for fairness.
## 1. Logistic Regression (Baseline Model)
In this step, a Logistic Regression model was implemented as the baseline classifier. The model was built using a pipeline that integrates the preprocessing stage with the classifier, ensuring that all transformations are applied consistently during training. The class_weight='balanced' option was used to address class imbalance, and the number of iterations was increased to ensure model convergence. The model was then trained on the training dataset.
```
log_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
log_reg.fit(X_train, y_train)
```

<img width="361" height="208" alt="PipelineRegression" src="https://github.com/user-attachments/assets/917978d3-765f-47db-8086-33d955353ef0" /> <br> Figure 17: PipelineRegression

This pipeline uses a ColumnTransformer to apply different preprocessing steps to different subsets of the data (likely numerical and categorical features) in parallel. The processed data is then combined and used to train a LogisticRegression model.

## 2. Decision Tree Classifier
In this step, a Decision Tree classifier was implemented to capture non-linear relationships in the data. The model was constructed using a pipeline that combines preprocessing with the classifier to ensure consistent data handling. Class imbalance was addressed using class_weight='balanced', model complexity was controlled through a maximum tree depth, and a fixed random state was applied for reproducibility. The model was trained using the training dataset.
```
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        max_depth=6
    ))
])

dt_model.fit(X_train, y_train)
```

<img width="364" height="210" alt="Pipeline DecisionTree" src="https://github.com/user-attachments/assets/c9ac59e1-33a9-408e-ae8b-af6a3e37e6bd" /> <br> Figure 18: Pipeline DecisionTree

The workflow utilizes a ColumnTransformer to apply specific preprocessing steps in parallel to different data types (numerical and categorical). The processed data is then fed into a DecisionTreeClassifier model.

## 3. Random Forest Classifier
In this step, a Random Forest classifier was implemented to improve predictive performance by leveraging an ensemble of decision trees. The model was built within a pipeline that integrates preprocessing and modeling to ensure consistent feature transformations. Class imbalance was handled using class_weight='balanced', while the number of trees and maximum depth were tuned to balance model complexity and generalization. The model was trained on the training dataset.
```
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        max_depth=10
    ))
])

rf_model.fit(X_train, y_train)
```

<img width="364" height="212" alt="Pipeline RandomForest" src="https://github.com/user-attachments/assets/d36c21ca-0e02-426f-8973-4da9a0f81692" /> <br> Figure 19: Pipeline RandomForest

The workflow utilizes a ColumnTransformer to apply specific preprocessing steps in parallel to different data types (numerical and categorical). The processed data is then fed into a RandomForestClassifier model.
Three machine learning models were trained to predict the probability of insurance claims:
-  Logistic Regression was used as a baseline model
-  Decision Tree captured non-linear patterns
-  Random Forest leveraged ensemble learning to improve predictive performance

Class weights were applied to address class imbalance.

# STEP 8: MODEL EVALUATION
## 1. Create an Evaluation Function (Clean & Reusable)
In this step, a reusable evaluation function was created to assess model performance on the test dataset. The function calculates key classification metrics including accuracy, precision, recall, F1-score, and ROC-AUC, and also outputs the confusion matrix. This allows consistent, repeatable evaluation for multiple models while providing a clear overview of predictive performance.
```
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
print(f"📌 {model_name} Performance")
    print("-" * 40)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    
print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
 ```
## 2. Evaluate Logistic Regression
In this step, the Logistic Regression baseline model was evaluated on the test dataset using the reusable evaluate_model function. Key performance metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix were calculated to assess how well the model predicts insurance claims. This provides a benchmark for comparing more complex models.  
```
evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
```
<img width="417" height="249" alt="Regression Performance" src="https://github.com/user-attachments/assets/b547390c-e123-49fc-b450-22ca4d7e879b" /> <br> Figure 20: Regression Performance

The Logistic Regression model exhibits moderate overall performance but struggles with precision, indicating a high rate of false positives. <br>
The confusion matrix [[812 293], [160 167]] provides a detailed breakdown of predictions: The high number of false positives (293) relative to true positives (167) confirms the low precision score. The model is cautious about predicting the positive class correctly but often flags too many instances incorrectly.

## 3. Evaluate Decision Tree
In this step, the Decision Tree model was evaluated on the test dataset using the evaluate_model function. Metrics including accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix were computed to measure the model’s ability to predict insurance claims. This evaluation helps compare its performance against the Logistic Regression baseline and assess the impact of capturing non-linear relationships.
```
evaluate_model(dt_model, X_test, y_test, "Decision Tree")
```

<img width="417" height="246" alt="Decision Tree Performance" src="https://github.com/user-attachments/assets/91c5ff50-55ce-4a74-9065-34216301d93a" /> <br> Figure 21: Decision Tree Performance

The Decision Tree model has moderate overall accuracy but poor precision, indicating that when it predicts the positive class, it is often incorrect. <br>
The confusion matrix [[684 421], [120 207]] provides a detailed breakdown of predictions: The large number of false positives (421) relative to true positives (207) confirms the low precision score. The model often predicts the positive class incorrectly, leading to many false alarms.

## 4. Evaluate Random Forest
In this step, the Random Forest model was evaluated on the test dataset using the evaluate model function. Key metrics—accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix were calculated to assess the model’s predictive performance. This evaluation allows comparison with the Logistic Regression and Decision Tree models, highlighting the benefits of ensemble learning in handling complex patterns and improving classification of insurance claims.
```
evaluate_model(rf_model, X_test, y_test, "Random Forest")
```

<img width="416" height="245" alt="Random Forest Performance" src="https://github.com/user-attachments/assets/94f2b8cd-eb00-4d82-b161-c5eac533d977" /> <br> Figure 22: Random Forest Performance

The Random Forest model has moderate accuracy but low precision, indicating that when it predicts the positive class, it is often incorrect.
This means the model is decent at overall classification but generates many false alarms for positive cases.
The model correctly predicted the positive class. In this case, 160 instances were correctly identified as having a claim. The model correctly predicted the negative class. Here, 849 instances were correctly identified as not having a claim. The model incorrectly predicted the positive class when it was actually negative (Type I error, a "false alarm"). The model predicted 256 claims that didn't actually occur.The model incorrectly predicted the negative class when it was actually positive (Type II error, a "missed opportunity"). The model missed 167 actual claims.

## 5. ROC Curve
In this step, ROC curves were plotted for selected models (Logistic Regression and Random Forest) to visualize their trade-off between true positive rate and false positive rate. ROC curves provide an intuitive way to compare classifier performance, assess discrimination ability, and complement numerical metrics like ROC-AUC. This step helps in understanding how well each model separates classes across different threshold settings.
```
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("ROC Curve - Logistic Regression")
plt.show()

RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("ROC Curve - Random Forest")
plt.show()
```

<img width="282" height="273" alt="ROC Curve Regression" src="https://github.com/user-attachments/assets/2642c098-fe19-46a3-9692-c1e59d363e3e" /> <br> Figure 23: ROC Curve Regression

The figure provided is a Receiver Operating Characteristic (ROC) curve for a logistic regression model, which graphically illustrates the model's performance at distinguishing between two classes. 
- AUC Score: The Area Under the Curve (AUC) is 0.68. An AUC score between 0.5 and 0.7 indicates a weak to acceptable ability to distinguish between classes, meaning the model is better than random guessing (AUC 0.5) but has limited predictive power for real-world application on its own.
- Curve Shape: The curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity). A perfect model's curve would hug the top-left corner, reaching a true positive rate of 1.0 while maintaining a false positive rate of 0.0. This curve, while better than the diagonal random guessing line, is relatively close to the center, reinforcing the moderate performance indicated by the AUC score.
- Trade-off: The curve shows the trade-off between sensitivity and specificity at different decision thresholds. As the model's ability to identify more true positives increases (moving up the y-axis), it also incorrectly flags more false positives (moving right on the x-axis). 

<img width="286" height="273" alt="ROC Curve RandomForest" src="https://github.com/user-attachments/assets/a315117d-d10a-495e-b81c-d729c2e67628" /> <br> Figure 24: ROC Curve RandomForest

The figure provided is a Receiver Operating Characteristic (ROC) curve for a Random Forest model, which graphically illustrates the model's ability to distinguish between two classes.
- AUC Score: The Area Under the Curve (AUC) is 0.67. An AUC score between 0.5 and 0.7 indicates a weak to acceptable ability to distinguish between classes, meaning the model is better than random guessing (AUC 0.5) but has limited predictive power for real-world application on its own.
- Curve Shape: The curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 - specificity). A perfect model's curve would hug the top-left corner. This curve, while better than the diagonal random guessing line, suggests only moderate performance.
- Trade-off: The curve shows the trade-off between identifying more true positives and incorrectly flagging more false positives at different decision thresholds.

# STEP 9: MODEL COMPARISON & FINAL SELECTION
## 1 Create a Model Comparison Table
In this step, the performance of all three models: Logistic Regression, Decision Tree, and Random Forest were compared using a summary table of key evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC. This table provides a clear, side-by-side view of each model’s predictive ability, facilitating an informed decision for final model selection based on overall performance and suitability for predicting insurance claims.

```
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, log_reg.predict(X_test)),
        accuracy_score(y_test, dt_model.predict(X_test)),
        accuracy_score(y_test, rf_model.predict(X_test))
    ],
    "Precision": [
        precision_score(y_test, log_reg.predict(X_test)),
        precision_score(y_test, dt_model.predict(X_test)),
        precision_score(y_test, rf_model.predict(X_test))
    ],
    "Recall": [
        recall_score(y_test, log_reg.predict(X_test)),
        recall_score(y_test, dt_model.predict(X_test)),
        recall_score(y_test, rf_model.predict(X_test))
    ],
    "F1-Score": [
        f1_score(y_test, log_reg.predict(X_test)),
        f1_score(y_test, dt_model.predict(X_test)),
        f1_score(y_test, rf_model.predict(X_test))
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]),
        roc_auc_score(y_test, dt_model.predict_proba(X_test)[:,1]),
        roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
    ]
})
```

### results

<img width="454" height="214" alt="Model Comparison" src="https://github.com/user-attachments/assets/5065fa82-2f52-48e5-90db-b6620b76a0fa" /> <br> Figure 25: Model Comparison


## Author
Monday Olawale
