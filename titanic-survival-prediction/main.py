"""Titanic survival prediction example.

This script loads the Titanic training data, adds a few simple engineered
features, prepares the data, trains a classifier, and prints evaluation
metrics.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Survived"
REQUIRED_COLUMNS = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name", TARGET_COLUMN}


def resolve_data_file() -> Path:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    # Check common Titanic dataset file names in both project root and script dir.
    candidates = [
        script_dir / "train.csv",
        root_dir / "train.csv",
        root_dir / "tested.csv",
        script_dir / "tested.csv",
        root_dir / "test.csv",
        script_dir / "test.csv",
    ]

    for file_path in candidates:
        if file_path.exists():
            return file_path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find a Titanic dataset CSV file. Searched these locations:\n"
        f"{searched}"
    )


def load_data() -> pd.DataFrame:
    data_file = resolve_data_file()
    df = pd.read_csv(data_file)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_sorted = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing_sorted}. "
            f"Loaded file: {data_file}"
        )

    print(f"Loaded dataset: {data_file}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    titles = data["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
    data["Title"] = titles.replace(
        {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Rare",
            "Countess": "Rare",
            "Capt": "Rare",
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


def build_model() -> Pipeline:
    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Embarked", "Title"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def main() -> None:
    df = load_data()
    df = add_features(df)

    print("First 5 rows after feature engineering:")
    print(df.head())
    print()

    print("Missing values:")
    print(df.isnull().sum())
    print()

    feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    model = build_model()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print()

    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
