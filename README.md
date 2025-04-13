# Crime Statistics Analysis Project

## Overview
This project demonstrates the use of Large Language Models (LLMs) with LangChain to analyze crime statistics data. The application processes crime data from 1950 to 2015, helps with data cleaning, and provides advanced analytical capabilities powered by Google's Gemini model.

## Table of Contents
- [Installation Requirements](#installation-requirements)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Key Features](#key-features)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Data Cleaning](#data-cleaning)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)

## Installation Requirements
The following packages are needed to run this project:
```python
import os
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
```

## Project Structure
The project follows this logical structure:
1. **Data Loading**: Load crime statistics from 'reported.csv'
2. **API Configuration**: Set up Google Gemini API connection
3. **Data Exploration**: Analyze and summarize the dataset structure
4. **Data Processing**: Document-based extraction and cleaning
5. **Analysis Chains**: Create LLM-based analysis chains for crime data insights
6. **Agent Implementation**: Build a CrimeDataAgent to streamline interactions

## Dataset Description
The dataset contains crime statistics spanning from 1950 to 2015 with the following details:

- **Shape**: 66 rows × 21 columns
- **Time Period**: 1950-2015
- **Key Categories**:
  - Year
  - crimes.total
  - crimes.penal.code
  - crimes.person
  - Specific crime types (murder, assault, sexual.offenses, rape, etc.)
  - Property crimes (stealing.general, burglary, vehicle.theft, etc.)
  - Other crimes (fraud, narcotics, drunk.driving)
  - population

The dataset has some missing values (represented as NA) in certain categories:
- house.theft: 15 missing values (22.73%)
- vehicle.theft: 7 missing values (10.61%)
- out.of.vehicle.theft: 15 missing values (22.73%)
- shop.theft: 15 missing values (22.73%)
- narcotics: 4 missing values (6.06%)

## Key Features

### 1. Data Exploration
The project provides a comprehensive data introduction function that reports:
- Dataset dimensions
- Column names and data types
- Missing value analysis
- Sample rows for quick inspection

```python
def data_intro(df):
    # Creates a detailed profile of the dataset
    # Returns profile dictionary and prints formatted summary
    # ...
```

### 2. Data Cleaning
The system includes an automatic data cleaning process that:
- Identifies missing values
- Replaces NAs with column averages
- Reports on changes made
- Provides statistical comparisons before and after cleaning

### 3. LLM-Based Analysis
Leverages Google's Gemini model to:
- Answer inference questions about crime trends
- Provide detailed analyses of data quality issues
- Generate insights based on the available crime statistics

### 4. Crime Data Agent
A comprehensive agent that integrates all functionality:
```python
class CrimeDataAgent:
    def __init__(self, df):
        self.df = df
        self.cleaned_df = None
        self.changes_report = None
    
    def analyze(self, query):
        """Answer inference questions about the data"""
        return analyze_crime_data(query, self.df)
    
    def clean_data(self):
        """Clean data by replacing NA values with column averages"""
        # ...
    
    def compare_stats(self, columns=None):
        """Compare statistics before and after cleaning"""
        # ...
```

## Implementation Details

### Gemini Model Integration
The project uses Google's Gemini 2.0 Flash model for analysis:

```python
model_name = 'gemini-2.0-flash'

llm_model = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=api_key,
    temperature=0.0,
    convert_system_message_to_human=True
)
```

### Data Analysis Functions
The system implements specialized functions for crime data analysis:

```python
def analyze_crime_data(query, df):
    """Use LLM to analyze crime data and answer inference questions"""
    
    # Create a summary of the data for context
    data_summary = f"""
    This dataset contains crime statistics from {df['Year'].min()} to {df['Year'].max()}.
    It includes {df.shape[1]} columns with various crime categories and population data.
    Total records: {df.shape[0]}
    
    # ...additional summary details...
    """
    
    inference_template = """
    You are a data analyst specialized in crime statistics. Based on the provided dataset information, 
    answer the following query with detailed analysis.
    
    Dataset Summary:
    {data_summary}
    
    User Query: {query}
    
    # ...template continues...
    """
```

### Data Cleaning Process
The NA replacement function handles missing values appropriately:

```python
def replace_na_with_average(df):
    """Replace NA values in each column with the column average"""
    
    # Create a copy of the dataframe to avoid modifying the original
    cleaned_df = df.copy()
    
    # Track what changes were made
    changes_report = {}
    
    # Process each column
    for column in cleaned_df.columns:
        # Check if column has NA values and is numeric
        if cleaned_df[column].isna().any() and np.issubdtype(cleaned_df[column].dtype, np.number):
            # Calculate average excluding NA values
            avg_value = cleaned_df[column].mean()
            # Count NA values
            na_count = cleaned_df[column].isna().sum()
            # Replace NA with average
            cleaned_df[column].fillna(avg_value, inplace=True)
            # Record the change
            changes_report[column] = {
                "na_count": int(na_count),
                "replacement_value": float(avg_value),
                "percentage_affected": float(na_count / len(cleaned_df) * 100)
            }
    
    return cleaned_df, changes_report
```

## Usage Examples

### 1. Initializing the Agent
```python
# Load the dataset
df = pd.read_csv('reported.csv')

# Initialize the agent
crime_agent = CrimeDataAgent(df)
```

### 2. Analyzing Crime Trends
```python
# Ask an inference question about violent crime trends
inference_result = crime_agent.analyze("What are the trends in violent crimes over time?")
print("INFERENCE ANALYSIS:")
print(inference_result)
```

Example output shows a comprehensive analysis of violent crime trends, including:
- Analysis of crimes.person, murder, assault, and rape over time
- Relationship to population changes
- Correlation analysis between crime types
- Limitations of the dataset and suggestions for deeper analysis

### 3. Cleaning the Data
```python
# Clean the data by replacing NA values with averages
changes = crime_agent.clean_data()
print("\nCLEANING REPORT:")
for col, details in changes.items():
    print(f"- {col}: Replaced {details['na_count']} NA values with {details['replacement_value']:.2f}")
```

Sample cleaning report:
```
CLEANING REPORT:
- house.theft: Replaced 15 NA values with 210.53
- vehicle.theft: Replaced 7 NA values with 466.29
- out.of.vehicle.theft: Replaced 15 NA values with 1192.57
- shop.theft: Replaced 15 NA values with 540.29
- narcotics: Replaced 4 NA values with 386.63
```

### 4. Comparing Statistics Before and After Cleaning
```python
# Compare statistics before and after cleaning
stats_comparison = crime_agent.compare_stats()
print("\nSTATISTICS COMPARISON:")
for col, stats in stats_comparison.items():
    print(f"- {col}:")
    print(f"  Before: mean={stats['original_mean']:.2f}, std={stats['original_std']:.2f}")
    print(f"  After:  mean={stats['cleaned_mean']:.2f}, std={stats['cleaned_std']:.2f}")
```

Sample comparison output:
```
STATISTICS COMPARISON:
- house.theft:
  Before: mean=210.53, std=48.68
  After:  mean=210.53, std=42.69
- vehicle.theft:
  Before: mean=466.29, std=193.70
  After:  mean=466.29, std=182.97
- out.of.vehicle.theft:
  Before: mean=1192.57, std=432.54
  After:  mean=1192.57, std=379.36
- shop.theft:
  Before: mean=540.29, std=185.30
  After:  mean=540.29, std=162.52
- narcotics:
  Before: mean=386.63, std=307.13
  After:  mean=386.63, std=297.53
```

## Data Cleaning
The analysis of data quality issues identified several potential areas for improvement:

1. **Missing Values**: Several fields have "NA" values that need proper handling
2. **Data Type Issues**: Ensuring consistent numeric types after handling missing values
3. **Consistency/Validity**: Verifying that crime category sums align with total crimes
4. **Column Naming**: Considering renaming columns to use underscores instead of periods

The implemented solution handles these issues by:
- Replacing NA values with column means for numeric fields
- Maintaining data type consistency
- Providing before/after statistical comparisons
- Documenting all changes made during cleaning

## Limitations
1. The system currently only implements mean imputation for missing values
2. no reasearch, can get API for the browser
3. prompt matters, i wrote 2 prommpt chains, while they are doing differnt things, procces differs in the same step, both of them had to do infernce step

