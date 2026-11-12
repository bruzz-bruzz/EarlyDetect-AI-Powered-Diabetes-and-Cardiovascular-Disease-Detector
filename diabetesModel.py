import pandas as pd
import joblib
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    precision_score,
)
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.025,
    max_depth=3,
    subsample=1.0,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
)
data = pd.read_csv("data.csv")
x = data
x = x.drop(columns=["diabetes"])
x["gender"] = x["gender"].map({"Male": 0, "Female": 1, "Other": 2})
x["smoking_history"] = x["smoking_history"].map(
    {"not current": 0, "former": 1, "never": 2, "current": 3, "No Info": 4, "ever": 5}
)
y = data["diabetes"]
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
undersampled = RandomUnderSampler(sampling_strategy=0.33, random_state=42)
x_train, y_train = undersampled.fit_resample(x_train, y_train)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
rec = recall_score(y_test, pred)
bal = balanced_accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
print("ACCURACY: " + str(acc))
print("F1 SCORE: " + str(f1))
print("RECALL SCORE: " + str(rec))
print("BALANCED ACCURACY SCORE: " + str(bal))
print("PRECISION SCORE: " + str(prec))
joblib.dump(model, "diabetesPredictor.pkl")