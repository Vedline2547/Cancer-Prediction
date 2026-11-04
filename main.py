# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\HP\Desktop\New ML\Cancer-prediction\cancer-risk-factors.csv")
# Few rows of dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())
# EDA 1
risk_counts = df['Risk_Level'].value_counts() / len(df) * 100
print(risk_counts)
df['Risk_Level'].value_counts().plot(kind='bar')
plt.show()
# EDA 2
print(df.Gender.value_counts())
print(df.Cancer_Type.value_counts())
df.Cancer_Type.value_counts().plot(kind='bar')
plt.show()
# Combining Gender and Caner Type
gender_cancer_counts = df.groupby(['Gender','Cancer_Type']).size().reset_index(name='Patient_Count')
gender_cancer_counts['Gender'] = gender_cancer_counts['Gender'].map({0:'Female',1:'Male'})
print(gender_cancer_counts)
sns.barplot(data=gender_cancer_counts,x='Cancer_Type',y='Patient_Count',hue='Gender')
plt.title('Cancer Type by Gender')
plt.xticks(rotation=45)
plt.show()
# EDA 3
# Find all patients under 35
young_patients = df[df['Age'] < 35]
print(young_patients)
count_young = (df['Age'] < 35).sum()
print(count_young)
young_cancer_types = df[df['Age'] < 35]['Cancer_Type'].value_counts()
print(young_cancer_types)
# EDA 4
# Finding out which factors highly contribute to high risk level
# Select only numerical columns
numeric_columns = df.select_dtypes(include='number').columns
risk_means = df.groupby('Risk_Level')[numeric_columns].mean().T 
print(risk_means)
# Focus on difference between High and Low Risk
risk_diff = risk_means['High'] - risk_means['Low']
risk_diff = risk_diff.sort_values(ascending=False)
print(risk_diff)
risk_diff.sort_values(ascending=True).plot(kind='barh',figsize=(8,6),title='Top Factors Contributing to High Risk')
plt.xlabel('Difference (High-Level)')
plt.ylabel('Feature')
plt.show()
bmi_by_risk = df.groupby('Risk_Level')['BMI'].mean()
print(bmi_by_risk)
# EDA 5
# Univariate Analysis
for i,predictor in enumerate(df.drop(columns=['Patient_ID','Risk_Level','Age','Overall_Risk_Score','BMI'])):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=predictor, hue='Risk_Level')
    plt.title(f'{predictor} vs Risk Level')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
# EDA 6
df = df.drop(columns=['Patient_ID'])
df['Risk_Level'] = df['Risk_Level'].map({'Low':0,'Medium':1,'High':2})
print(df.head())
df_dummies = pd.get_dummies(df)
print(df_dummies.head())
plt.figure(figsize=(20,8))
df_dummies.corr()['Risk_Level'].sort_values(ascending=False).plot(kind='bar')
plt.show()