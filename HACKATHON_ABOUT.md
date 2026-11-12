## Inspiration
Diabetes and cardiovascular diseases are extremely common problems worldwide, primarily affecting developed nations. For example, diabetes affects one out of nine people, and around 43% of people with diabetes have it undiagnosed, causing 3.4 million deaths in 2024. On the other hand, cardiovascular diseases cause roughly 20 million deaths a year, whereas 620 million people suffer from cardiovascular diseases. This project aims to make early detection and diagnosis of these fatal diseases more accessible for the general public and medical personnel. 
## What it does
Utilizes a classification model (XGBoost or CatBoost, depending on the disease) to determine if someone is at risk of a specific disease. Additionally, the app utilizes SHAP to improve explainability and interpretability of the results.
## How we built it
Built using XGBoost and CatBoost classification models for the diabetes and cardiovascular disease models, respectively. The frontend was built using Streamlit with Python to act as a user-friendly and simple interface to input their biometrics and receive detailed predictions and SHAP explanations.
## Challenges we ran into
The dataset we used for the diabetes model was moderately imbalanced, with the ratio of diabetic and non-diabetic cases being around 11:1. To solve this, we utilized `RandomUnderSampling` to bring the ratio down to a manageable ratio of 3:1, which helped in ensuring model robustness and preventing bias.
## Accomplishments that we're proud of
Our models achieved outstanding performance in a range of metrics. The metrics collected were accuracy score, precision score, recall score, balanced accuracy score, and F1 score. All values are rounded to 3 decimal places.

| Model | Accuracy | Precision | Recall | Balanced Accuracy | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Diabetes (XGBoost)** | 0.959 | 0.752 | 0.787 | 0.881 | 0.769 |
| **Cardiovascular Disease (CatBoost)** | 0.705 | 0.663 | 0.833* | 0.705 | 0.738 |

*\*Recall was prioritized in the Cardiovascular model to minimize false negatives.*

## What we learned
Class imbalances in datasets are quite common, so undersampling and other techniques such as SMOTE are crucial for handling such datasets.
## What's next for Diabetes and Cardiovascular Disease Predictor
- Adding more explainability features other than SHAP
- Expanding the app to cover a wider range of diseases
- Integrating with electronic health records (EHR) for medical personnel