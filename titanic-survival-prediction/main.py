"""Titanic survival prediction example.

This script loads the Titanic training data, adds a few simple engineered
features, prepares the data, trains a classifier, and prints evaluation
metrics.
"""

from pathlib import Path

import pandas as pd
# sklearn imports for machine learning pipeline components
from sklearn.compose import ColumnTransformer                                                # Combine multiple transformers for different feature types
from sklearn.impute import SimpleImputer                                                  # Handle missing values in the data
from sklearn.linear_model import LogisticRegression                                  # Classification model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix                # Model evaluation metrics
from sklearn.model_selection import train_test_split                                        # Split data into training and test sets
from sklearn.pipeline import Pipeline                                                     # Chain transformers and models together
from sklearn.preprocessing import OneHotEncoder, StandardScaler                      # Encode categorical features and scale numeric features


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Target column that we want to predict (whether passenger survived or not)
TARGET_COLUMN = "Survived"

# Set of required columns that must exist in the dataset for the script to work
REQUIRED_COLUMNS = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", TARGET_COLUMN}


# ============================================================================
# DATA LOADING & FILE RESOLUTION
# ============================================================================

def resolve_data_file() -> Path:
    """
    Locate the Titanic dataset CSV file.
    
    Searches for 'tested.csv' in two locations:
    1. The same directory as this script
    2. The parent directory (project root)
    
    Returns:
        Path: Absolute path to the found CSV file
        
    Raises:
        FileNotFoundError: If the dataset file is not found in either location
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    # Get the parent directory (one level up)
    root_dir = script_dir.parent

    # Define search locations in priority order (script dir first, then root)
    candidates = [
        script_dir / "tested.csv",
        root_dir / "tested.csv",
    ]

    # Search for the file in each candidate location
    for file_path in candidates:
        if file_path.exists():
            return file_path

    # If file not found, provide helpful error message with all searched locations
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a Titanic dataset CSV file. Searched these locations:\n"
        f"{searched}"
    )

#data loading & validation....
def load_data() -> pd.DataFrame:
    """
    Load the Titanic dataset from CSV and validate it has required columns.
    
    This function:
    1. Finds the dataset file using resolve_data_file()
    2. Reads the CSV file into a pandas DataFrame
    3. Checks that all required columns are present
    4. Raises an error if any required columns are missing
    
    Returns:
        pd.DataFrame: The loaded Titanic dataset
        
    Raises:
        ValueError: If any required columns are missing from the dataset
    """
    # Locate the dataset file
    data_file = resolve_data_file()
    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_file)

    # Find any required columns that are missing from the dataset
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    # If columns are missing, raise an error with details
    if missing_columns:
        missing_sorted = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing_sorted}. "
            f"Loaded file: {data_file}"
        )

    # Confirm successful load with file path
    print(f"Loaded dataset: {data_file}")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

#feature engineering & model building....
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data to improve model performance.
    
    This function engineers three new features:
    1. FamilySize: Total family members (SibSp + Parch + 1, including self)
    2. IsAlone: Binary indicator if passenger was traveling alone (FamilySize == 1)
    3. Title: Extracted from passenger names (Mr, Mrs, Miss, Rare), helps capture social status
    
    Args:
        df (pd.DataFrame): Original dataset
        
    Returns:
        pd.DataFrame: Dataset with new engineered features added
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()

    # FEATURE 1: Calculate total family size
    # SibSp = number of siblings/spouses
    # Parch = number of parents/children
    # Add 1 to include the passenger themselves
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    
    # FEATURE 2: Create binary indicator for traveling alone
    # 1 if FamilySize equals 1, otherwise 0
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # FEATURE 3: Extract title (Mr., Mrs., Miss., etc.) from passenger names
    # Regex pattern: Find text between comma and period (e.g., "Smith, Mr." -> "Mr")
    titles = data["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    
    # Standardize titles - group rare/uncommon titles together
    # Maps various titles to standard categories: Mr, Mrs, Miss, and Rare
    data["Title"] = titles.replace(
        {
            "Mlle": "Miss",      # French "Miss"
            "Ms": "Miss",        # Modern "Ms"
            "Mme": "Mrs",        # French "Mrs"
            "Lady": "Rare",      # Nobility titles grouped as "Rare"
            "Countess": "Rare",
            "Capt": "Rare",      # Military titles grouped as "Rare"
            "Col": "Rare",
            "Don": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Rev": "Rare",
            "Sir": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare",
        }
    )

    return data


# ============================================================================
# MODEL BUILDING & DATA PREPROCESSING PIPELINE
# ============================================================================

#model building & evaluation....
def build_model() -> Pipeline:
    """
    Build a complete machine learning pipeline with preprocessing and classification.
    
    The pipeline consists of three main stages:
    1. Data Preprocessing:
       - Numeric features: Handle missing values (median) and scale
       - Categorical features: Handle missing values (most frequent) and encode
    2. Classification Model: Logistic Regression with balanced class weights
    
    Returns:
        Pipeline: A scikit-learn pipeline ready for training and prediction
    """
    # Define numeric feature columns that need scaling
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    
    # Define categorical feature columns that need encoding
    categorical_features = ["Sex", "Embarked", "Title"]

    # ===== NUMERIC TRANSFORMER =====
    # Pipeline for preprocessing numeric features
    numeric_transformer = Pipeline(
        steps=[
            # Step 1: Impute missing values with median (robust to outliers)
            ("imputer", SimpleImputer(strategy="median")),
            # Step 2: Scale features to have mean=0 and std=1 (important for Logistic Regression)
            ("scaler", StandardScaler()),
        ]
    )

    # ===== CATEGORICAL TRANSFORMER =====
    # Pipeline for preprocessing categorical features
    categorical_transformer = Pipeline(
        steps=[
            # Step 1: Impute missing values with most frequent category
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Step 2: One-Hot Encode categorical features (convert text to numeric)
            # handle_unknown="ignore" prevents errors when test data has unseen categories
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ===== COMBINE TRANSFORMERS =====
    # Use ColumnTransformer to apply different transformations to different feature types
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),     # Apply numeric transformer to numeric columns
            ("cat", categorical_transformer, categorical_features),  # Apply categorical transformer to categorical columns
        ]
    )

    # ===== COMPLETE PIPELINE =====
    # Combine preprocessing with the classification model
    return Pipeline(
        steps=[
            # Step 1: Preprocess all features
            ("preprocessor", preprocessor),
            # Step 2: Train Logistic Regression classifier
            # max_iter=1000: Allow more iterations for convergence
            # class_weight="balanced": Automatically adjust for imbalanced classes
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


# ============================================================================
# MAIN EXECUTION: TRAIN & EVALUATE MODEL
# ============================================================================

def main() -> None:
    """
    Main orchestration function that runs the complete ML workflow:
    1. Load and explore data
    2. Engineer features
    3. Build preprocessing pipeline and model
    4. Split data into training/test sets
    5. Train the model
    6. Evaluate performance with metrics
    """
    
    # ===== STEP 1: LOAD & EXPLORE DATA =====
    print("\n" + "="*70)
    print("STEP 1: Loading and Exploring Data")
    print("="*70)
    
    # Load the Titanic dataset with validation
    df = load_data()
    
    # Add engineered features (FamilySize, IsAlone, Title)
    df = add_features(df)

    # Display first 5 rows to verify data structure
    print("\nFirst 5 rows after feature engineering:")
    print(df.head())
    print()

    # Identify and display missing values for data quality assessment
    print("Missing values by column:")
    print(df.isnull().sum())
    print()

    # ===== STEP 2: PREPARE FEATURES & TARGET =====
    print("="*70)
    print("STEP 2: Preparing Features for Model")
    print("="*70)
    
    # Select feature columns (all except target column)
    feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
    X = df[feature_columns]  # Input features
    y = df[TARGET_COLUMN]     # Target variable (Survived: 0 or 1)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Survival rate in dataset: {y.mean():.2%}")
    print()

    # ===== STEP 3: BUILD MODEL & PIPELINE =====
    print("="*70)
    print("STEP 3: Building ML Pipeline")
    print("="*70)
    
    # Create the complete preprocessing and classification pipeline
    model = build_model()
    print("\nPipeline created successfully!")
    print()

    # ===== STEP 4: SPLIT DATA =====
    print("="*70)
    print("STEP 4: Splitting Data into Training & Test Sets")
    print("="*70)
    
    # Split data: 80% training, 20% testing
    # random_state=42 ensures reproducibility (same split each run)
    # stratify=y ensures both train/test have similar survival rates
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,          # 20% for testing
        random_state=42,        # Reproducible random split
        stratify=y,             # Maintain class distribution
    )
    
    print(f"\nTraining rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print()

    # ===== STEP 5: TRAIN MODEL =====
    print("="*70)
    print("STEP 5: Training the Model")
    print("="*70)
    
    # Fit the model on training data
    # This trains the pipeline: preprocessing transforms the data, then Logistic Regression learns patterns
    model.fit(X_train, y_train)
    print("\nModel training completed!")
    print()

    # ===== STEP 6: MAKE PREDICTIONS & EVALUATE =====
    print("="*70)
    print("STEP 6: Evaluating Model Performance")
    print("="*70)
    
    # Generate predictions on test data
    predictions = model.predict(X_test)

    # Calculate accuracy (percentage of correct predictions)
    accuracy = accuracy_score(y_test, predictions)

    # Print summary statistics
    print(f"\nTest Set Size: {len(X_test)} passengers")
    print(f"Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()

    # Display Confusion Matrix
    # True Negatives (TN) | False Positives (FP)
    # False Negatives (FN) | True Positives (TP)
    print("Confusion Matrix:")
    print("(Shows True vs Predicted - Rows: Actual, Columns: Predicted)")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print()

    # Display Classification Report
    # Shows Precision, Recall, F1-Score for each class (Did Not Survive vs Survived)
    print("Classification Report:")
    print("(Precision: accuracy of positive predictions)")
    print("(Recall: % of actual positives correctly identified)")
    print("(F1-Score: harmonic mean of precision and recall)")
    print(classification_report(y_test, predictions, zero_division=0))
    
    print("="*70)
    print("Model evaluation completed!")
    print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
