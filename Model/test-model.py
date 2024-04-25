import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
pd.set_option('display.max_columns', None)

loaded_model = joblib.load('Model/model.pkl')
loaded_scaler = joblib.load('Model/scaler.pkl')

data = pd.read_csv('Model/data.csv')
data_scaled = loaded_scaler.transform(data.drop('Target', axis=1))

predictions = loaded_model.predict(data_scaled)
accuracy = accuracy_score(data['Target'], predictions)

coefficients = loaded_model.coef_[0]
features = loaded_scaler.feature_names_in_

feature_importance = pd.Series(coefficients, index=features).sort_values(ascending=False)

# print(feature_importance)
# print(accuracy)
