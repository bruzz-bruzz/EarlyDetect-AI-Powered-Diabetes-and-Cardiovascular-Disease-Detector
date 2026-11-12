import joblib
model = joblib.load('diabetesPredictor.pkl')
print(dir(model))
if hasattr(model, 'feature_names_in_'):
    print(model.feature_names_in_)
else:
    print("No feature_names_in_ attribute")
