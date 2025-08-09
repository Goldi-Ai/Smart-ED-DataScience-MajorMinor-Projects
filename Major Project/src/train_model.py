import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from data_prep import load_data, prepare_train_test, save_preprocessor

# ==== CONFIG ====
TARGET_COL = "Default"

# ==== FIXED PATHS ====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Project root
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==== PLOTTING FUNCTIONS ====
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

def plot_roc_curve(y_true, y_probs, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve: {save_path}")

# ==== MAIN TRAINING ====
def main():
    print("Loading data...")
    df = load_data()

    print("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor, num_cols, cat_cols = prepare_train_test(df, target_col=TARGET_COL)

    # Save preprocessor
    save_preprocessor(preprocessor, path=os.path.join(MODELS_DIR, "preprocessor.pkl"))

    # Models to try
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    results = {}
    best_model = None
    best_score = 0

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)

        results[name] = {"Accuracy": acc, "F1": f1, "ROC_AUC": roc_auc}

        # Save plots
        plot_confusion_matrix(y_test, y_pred, f"{name} Confusion Matrix", f"{name}_cm.png")
        plot_roc_curve(y_test, y_probs, f"{name} ROC Curve", f"{name}_roc.png")

        # Track best
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    print("\nModel performance:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    # Save best model
    best_model_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model saved to {best_model_path}")

    # Feature importance if available
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(importances)), importances)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
        plt.close()
        print(f"Saved feature importance: {os.path.join(PLOTS_DIR, 'feature_importance.png')}")

if __name__ == "__main__":
    main()
