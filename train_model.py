import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_URL = "https://raw.githubusercontent.com/Zue77/Pima-Indians-Diabetes-Dataset/main/diabetes.csv"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def main():
    csv_path = Path("diabetes.csv")
    if csv_path.exists():
        data = pd.read_csv(csv_path)
    else:
        print("diabetes.csv not found. Downloading the public Pima dataset...")
        data = pd.read_csv(DATA_URL)
        data.to_csv(csv_path, index=False)
        print("Dataset saved as diabetes.csv")

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset columns missing: {missing}")

    data = data[required].copy()
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, float("nan"))
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    print("\nMODEL EVALUATION (20% hold-out test set)")
    print(f"Accuracy : {accuracy_score(y_test, predicted):.3f}")
    print(f"Precision: {precision_score(y_test, predicted, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predicted, zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, predicted, zero_division=0):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probability):.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, predicted))

    joblib.dump(model, "diabetes_model.pkl")
    print("\nDone! Created diabetes_model.pkl")

if __name__ == "__main__":
    main()
