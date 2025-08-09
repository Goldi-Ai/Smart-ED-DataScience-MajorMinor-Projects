import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def load_data(path=None):
    """
    Load CSV into a pandas DataFrame.
    If no path is given, loads from ../data/Loan_default.csv (relative to project root).
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
        path = os.path.join(base_dir, "data", "Loan_default.csv")
    df = pd.read_csv(path)
    return df

def basic_report(df, n=5):
    """
    Print basic info about the dataframe for quick EDA.
    """
    print("Shape:", df.shape)
    display_df = df.head(n)
    print("\n--- HEAD ---")
    print(display_df.to_string(index=False))
    print("\n--- INFO ---")
    print(df.info())
    print("\n--- Describe (numeric) ---")
    print(df.describe().T)
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

def build_preprocessing_pipeline(df, target_col="Default", high_card_threshold=50):
    """
    Build preprocessing pipeline with OneHot for low-card categorical and frequency encoding for high-card.
    """
    # Drop ID-like columns
    id_like_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_like_cols)

    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Split categorical into low-cardinality and high-cardinality
    low_card_cols = [col for col in categorical_cols if df[col].nunique() <= high_card_threshold]
    high_card_cols = [col for col in categorical_cols if df[col].nunique() > high_card_threshold]

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # One-hot encoding for low-card categorical
    low_card_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Frequency encoding for high-card categorical
    def freq_encode_column(col):
        freq = col.value_counts(normalize=True)
        return col.map(freq)

    # Apply frequency encoding before pipeline (manual step)
    for col in high_card_cols:
        X[col] = freq_encode_column(X[col])

    # Add high-card columns to numeric processing
    numeric_cols += high_card_cols

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', low_card_transformer, low_card_cols)
        ],
        remainder='drop'
    )

    return preprocessor, numeric_cols, low_card_cols, id_like_cols

def prepare_train_test(df, target_col="Default", test_size=0.2, random_state=42):
    """
    Prepare train-test split and preprocessing.
    """
    y = df[target_col].copy()
    if y.dtype == 'object' or y.dtype.name == 'category':
        try:
            y = y.astype(int)
        except:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        y = y.astype(int)

    preprocessor, numeric_cols, low_card_cols, dropped_cols = build_preprocessing_pipeline(df, target_col=target_col)

    X = df.drop(columns=[target_col] + dropped_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor, numeric_cols, low_card_cols

def save_preprocessor(preprocessor, path=None):
    """
    Save the fitted preprocessor.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")

def load_preprocessor(path=None):
    """
    Load the preprocessor.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base_dir, "models", "preprocessor.pkl")
    return joblib.load(path)
