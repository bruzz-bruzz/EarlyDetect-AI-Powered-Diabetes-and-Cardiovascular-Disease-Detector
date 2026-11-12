import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)
import pandas as pd
data = pd.read_csv("data.csv", index_col=False, sep=";")
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.06,
    depth=5,
    subsample=0.8,
    colsample_bylevel=0.8,
    auto_class_weights="Balanced",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=False,
)
x = data
x = x.drop(columns=["cardio", "id"])
y = data["cardio"]
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], use_best_model=True)
probs = model.predict_proba(x_test)[:, 1]
pred = (probs >= 0.35).astype(int)
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
bae = balanced_accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
print("ACCURACY SCORE: ", acc)
print("PRECISION SCORE: ", prec)
print("RECALL SCORE: ", rec)
print("BALANCED ACCURACY SCORE: ", bae)
print("F1 SCORE: ", f1)
joblib.dump(model, "heartPredictor.pkl")
