"""
Download Financial Risk Assessment Dataset from Kaggle
"""
import os


def download_data():
    """Download dataset from Kaggle using kagglehub."""
    try:
        import kagglehub

        print("Downloading Financial Risk Assessment Dataset...")

        # Download the dataset
        path = kagglehub.dataset_download("lorenzozoppelletto/financial-risk-for-loan-approval")

        print(f"Dataset downloaded to: {path}")

        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        # Copy CSV file to data directory
        import shutil

        for file in os.listdir(path):
            if file.endswith(".csv"):
                src = os.path.join(path, file)
                dst = os.path.join(data_dir, file)
                shutil.copy(src, dst)
                print(f"Copied {file} to {data_dir}")

        print("✓ Download complete!")
        return data_dir

    except ImportError:
        print("kagglehub not installed. Install with: pip install kagglehub")
        raise
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Try alternative: create sample data for CI
        print("Creating sample data for testing...")
        create_sample_data()


def create_sample_data():
    """Create sample data for CI/testing when Kaggle download fails."""
    import random

    data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Create sample CSV
    csv_path = os.path.join(data_dir, "financial_risk.csv")

    headers = [
        "Age",
        "Gender",
        "Education Level",
        "Marital Status",
        "Income",
        "Credit Score",
        "Loan Amount",
        "Loan Purpose",
        "Employment Status",
        "Years at Current Job",
        "Payment History",
        "Debt-to-Income Ratio",
        "Assets Value",
        "Number of Dependents",
        "City",
        "State",
        "Country",
        "Previous Defaults",
        "Marital Status Change",
        "Risk Rating",
    ]

    genders = ["Male", "Female"]
    education = ["High School", "Bachelor's", "Master's", "PhD"]
    marital = ["Single", "Married", "Divorced"]
    purposes = ["Auto", "Home", "Education", "Personal"]
    employment = ["Employed", "Self-Employed", "Unemployed"]
    payment = ["Good", "Fair", "Poor"]
    risks = ["Low", "Medium", "High"]

    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")

        for _ in range(1000):
            row = [
                random.randint(22, 65),  # Age
                random.choice(genders),
                random.choice(education),
                random.choice(marital),
                random.randint(30000, 150000),  # Income
                random.randint(550, 800),  # Credit Score
                random.randint(5000, 100000),  # Loan Amount
                random.choice(purposes),
                random.choice(employment),
                random.randint(0, 20),  # Years at job
                random.choice(payment),
                round(random.uniform(0.1, 0.6), 2),  # DTI
                random.randint(10000, 500000),  # Assets
                random.randint(0, 4),  # Dependents
                "City",
                "State",
                "Country",
                random.randint(0, 2),  # Defaults
                random.randint(0, 2),  # Marital change
                random.choice(risks),
            ]
            f.write(",".join(map(str, row)) + "\n")

    print(f"✓ Created sample data at {csv_path}")


if __name__ == "__main__":
    download_data()
