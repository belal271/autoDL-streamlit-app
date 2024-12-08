import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Handle NaN values
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Identify column types
    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
    boolean_cols = df_copy.select_dtypes(include=['bool']).columns
    
    # Handle NaN values in numerical columns
    if len(numerical_cols) > 0:
        df_copy[numerical_cols] = numerical_imputer.fit_transform(df_copy[numerical_cols])
    
    # Handle NaN values in categorical columns
    if len(categorical_cols) > 0:
        df_copy[categorical_cols] = categorical_imputer.fit_transform(df_copy[categorical_cols])
    
    # Convert boolean columns to integers (0 and 1)
    if len(boolean_cols) > 0:
        for col in boolean_cols:
            df_copy[col] = df_copy[col].astype(int)
    
    # Scale numerical columns
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    
    # Encode categorical columns
    if len(categorical_cols) > 0:
        le = LabelEncoder()
        for col in categorical_cols:
            df_copy[col] = le.fit_transform(df_copy[col])
    
    return df_copy
