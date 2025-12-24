-- Database Schema initialization for Financial Risk Prediction
-- Star Schema Design (Aligned with actual dataset)

-- Dimension: Customers
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id SERIAL PRIMARY KEY,
    age INT,
    gender VARCHAR(20),
    education_level VARCHAR(50),
    marital_status VARCHAR(30),
    income FLOAT,
    employment_status VARCHAR(50),
    years_at_current_job FLOAT,
    number_of_dependents INT
);

-- Dimension: Geography
CREATE TABLE IF NOT EXISTS dim_geography (
    geography_id SERIAL PRIMARY KEY,
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(100)
);

-- Dimension: Credit History
CREATE TABLE IF NOT EXISTS dim_credit_history (
    credit_history_id SERIAL PRIMARY KEY,
    credit_score FLOAT,
    payment_history VARCHAR(20),
    previous_defaults FLOAT
);

-- Fact Table: Loans
CREATE TABLE IF NOT EXISTS fact_loans (
    loan_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES dim_customers(customer_id),
    geography_id INT REFERENCES dim_geography(geography_id),
    credit_history_id INT REFERENCES dim_credit_history(credit_history_id),
    loan_amount FLOAT,
    loan_purpose VARCHAR(50),
    debt_to_income_ratio FLOAT,
    assets_value FLOAT,
    marital_status_change INT,
    risk_rating VARCHAR(20) -- Target Variable: Low, Medium, High
);
