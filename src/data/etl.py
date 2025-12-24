"""
ETL Script for Financial Risk Prediction Platform
Aligned with actual dataset: Age, Gender, Education Level, ... Risk Rating
"""
import os

import pandas as pd
from sqlalchemy import create_engine

# Database connection details
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "risk_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def run_etl():
    print("=" * 50)
    print("Starting ETL Process...")
    print("=" * 50)

    # 1. Read Data
    data_path = os.path.join(os.getcwd(), "data", "financial_risk_assessment.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    print(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Data Cleaning
    print("\n[Step 1] Cleaning data...")
    
    # Drop duplicates
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  - Removed {initial_count - len(df)} duplicate rows")

    # Standardize column names (remove spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    print(f"  - Standardized column names: {list(df.columns)}")

    # Handle Missing Values
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numerical_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  - Filled {missing} missing values in '{col}' with median")

    for col in categorical_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  - Filled {missing} missing values in '{col}' with mode")

    # 3. Transform into Star Schema
    print("\n[Step 2] Transforming into Star Schema...")

    # --- Dimension: Customers ---
    dim_customers = df[[
        'age', 'gender', 'education_level', 'marital_status', 
        'income', 'employment_status', 'years_at_current_job', 'number_of_dependents'
    ]].copy()
    dim_customers = dim_customers.drop_duplicates().reset_index(drop=True)
    dim_customers['customer_id'] = dim_customers.index + 1
    print(f"  - Created dim_customers with {len(dim_customers)} unique records")

    # --- Dimension: Geography ---
    dim_geography = df[['city', 'state', 'country']].copy()
    dim_geography = dim_geography.drop_duplicates().reset_index(drop=True)
    dim_geography['geography_id'] = dim_geography.index + 1
    print(f"  - Created dim_geography with {len(dim_geography)} unique records")

    # --- Dimension: Credit History ---
    dim_credit_history = df[['credit_score', 'payment_history', 'previous_defaults']].copy()
    dim_credit_history = dim_credit_history.drop_duplicates().reset_index(drop=True)
    dim_credit_history['credit_history_id'] = dim_credit_history.index + 1
    print(f"  - Created dim_credit_history with {len(dim_credit_history)} unique records")

    # --- Fact Table: Loans ---
    # We need to map each row to its dimension keys
    # For simplicity, we'll create surrogate keys based on merge
    
    # Map customer_id
    df = df.merge(
        dim_customers[['age', 'gender', 'education_level', 'marital_status', 
                       'income', 'employment_status', 'years_at_current_job', 
                       'number_of_dependents', 'customer_id']],
        on=['age', 'gender', 'education_level', 'marital_status', 
            'income', 'employment_status', 'years_at_current_job', 'number_of_dependents'],
        how='left'
    )

    # Map geography_id
    df = df.merge(
        dim_geography[['city', 'state', 'country', 'geography_id']],
        on=['city', 'state', 'country'],
        how='left'
    )

    # Map credit_history_id
    df = df.merge(
        dim_credit_history[['credit_score', 'payment_history', 'previous_defaults', 'credit_history_id']],
        on=['credit_score', 'payment_history', 'previous_defaults'],
        how='left'
    )

    fact_loans = df[[
        'customer_id', 'geography_id', 'credit_history_id',
        'loan_amount', 'loan_purpose', 'debt_to_income_ratio', 
        'assets_value', 'marital_status_change', 'risk_rating'
    ]].copy()
    fact_loans['loan_id'] = fact_loans.index + 1
    print(f"  - Created fact_loans with {len(fact_loans)} records")

    # 4. Load to PostgreSQL
    print("\n[Step 3] Loading data to PostgreSQL...")
    try:
        engine = create_engine(DATABASE_URL)

        # Reorder columns since ID is serial in DB, we skip it
        dim_customers_load = dim_customers.drop(columns=['customer_id'])
        dim_geography_load = dim_geography.drop(columns=['geography_id'])
        dim_credit_history_load = dim_credit_history.drop(columns=['credit_history_id'])
        fact_loans_load = fact_loans.drop(columns=['loan_id'])

        dim_customers_load.to_sql('dim_customers', engine, if_exists='append', index=False, method='multi', chunksize=500)
        print("  - Loaded dim_customers")

        dim_geography_load.to_sql('dim_geography', engine, if_exists='append', index=False, method='multi', chunksize=500)
        print("  - Loaded dim_geography")

        dim_credit_history_load.to_sql('dim_credit_history', engine, if_exists='append', index=False, method='multi', chunksize=500)
        print("  - Loaded dim_credit_history")

        fact_loans_load.to_sql('fact_loans', engine, if_exists='append', index=False, method='multi', chunksize=500)
        print("  - Loaded fact_loans")

        print("\n" + "=" * 50)
        print("ETL Process Completed Successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError loading data: {e}")
        raise


if __name__ == "__main__":
    run_etl()
