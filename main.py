import joblib
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
data = pd.read_csv('data.csv',index_col=False)
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42
)
x = data
x = x.drop(columns=['cardio','id'])
y = data['cardio']
x_train,x_temp,y_train,y_temp = train_test_split(x,y,test_size=0.3,random_state=42,stratify=y)
x_val,x_test,y_val,y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=False)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
prec = precision_score(y_test,pred)
rec = recall_score(y_test,pred)
cm = confusion_matrix(y_test, pred)
f1 = f1_score(y_test,pred)
print("ACCURACY SCORE: ", acc)
print("PRECISION SCORE: ", prec)
print("RECALL SCORE: ", rec)
print("CONFUSION MATRIX: \n",cm)
print("F1 SCORE: ", f1)
joblib.dump(model,'predictor.pkl')
explainer = shap.TreeExplainer(model)
shap_values = explainer(x_test)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
if hasattr(x_test, "columns"):
    feature_names = x_test.columns.tolist()
else:
    feature_names = [f"Feature_{i}" for i in range(x_test.shape[1])]

# 4. Create a clean, sorted DataFrame
shap_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Abs_SHAP": mean_abs_shap,
    "Relative_Importance_%": (mean_abs_shap / mean_abs_shap.sum()) * 100
}).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

# 5. Print the table and optionally save to CSV
print(shap_importance_df.to_string(index=False))