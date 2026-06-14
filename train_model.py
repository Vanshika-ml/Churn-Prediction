import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/telecom_churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.dropna(inplace=True)

encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col]
        )

        encoders[col] = le

X = df.drop("Churn", axis=1)

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print(
    f"Accuracy: {acc*100:.2f}%"
)

os.makedirs("models",exist_ok=True)

joblib.dump(
    model,
    "models/churn_model.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

print("Model Saved")