import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("macro_events.csv")

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("datetime")


def clean_numeric(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("M", "", regex=False)
        .str.replace("K", "", regex=False)
        .replace("", np.nan)
        .replace("nan", np.nan)
        .replace("-", np.nan)
        .astype(float)
    )

for col in ["Actual", "Forecast", "Previous"]:
    df[col] = clean_numeric(df[col])

df = df.dropna(subset=["Actual", "Forecast", "Previous"])

tolerance = 0.1 

diff = df["Actual"] - df["Forecast"]

def label_surprise(x):
    if x > tolerance:
        return 1   # Beat
    elif x < -tolerance:
        return -1  # Miss
    else:
        return 0   # Neutral

df["target"] = diff.apply(label_surprise)

df["abs_forecast"] = df["Forecast"].abs()

df["prev_change"] = df["Previous"].diff()

df = df.set_index("datetime")
df = df.groupby(["Currency", "Event"], group_keys=False).apply(
    lambda g: g.assign(
        rolling_actual_mean=g["Actual"].rolling(window=3, min_periods=1).mean(),
        rolling_forecast_mean=g["Forecast"].rolling(window=3, min_periods=1).mean(),
        rolling_error_mean=(g["Actual"] - g["Forecast"]).rolling(window=3, min_periods=1).mean(),
    )
)

impact_map = {"low": 0, "medium": 1, "high": 2}
df["Impact_num"] = df["Impact"].str.lower().map(impact_map)

df["year"] = df.index.year
df["month"] = df.index.month
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour

df = df.reset_index()


numeric_features = [
    "Actual", "Forecast", "Previous",
    "abs_forecast", "prev_change",
    "rolling_actual_mean", "rolling_forecast_mean", "rolling_error_mean",
    "Impact_num", "year", "month", "dayofweek", "hour"
]

categorical_features = ["Currency", "Event", "Country"]

X = df[numeric_features + categorical_features]
y = df["target"]


split_date = pd.Timestamp("2012-01-01")  # adjust based on your range
train_mask = df["datetime"] < split_date
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

ohe = clf.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

xgb_model = clf.named_steps["model"]
importances = xgb_model.feature_importances_

idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(8, 6))
plt.barh(np.array(all_feature_names)[idx][::-1], importances[idx][::-1])
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()
