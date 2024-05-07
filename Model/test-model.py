import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
pd.set_option('display.max_columns', None)

loaded_model = joblib.load('Model/model.pkl')
loaded_scaler = joblib.load('Model/scaler.pkl')

data = pd.read_csv('Model/dota2Test.csv')
data.columns = ['Target'] + ['Game mode'] + ['Game type'] + [f'Feature_{i}' for i in range(1, data.shape[1] - 2)]
data_scaled = loaded_scaler.transform(data.drop('Target', axis=1))

predictions = loaded_model.predict(data_scaled)
accuracy = accuracy_score(data['Target'], predictions)

coefficients = loaded_model.coef_[0]
features = loaded_scaler.feature_names_in_

feature_importance = pd.Series(coefficients, index=features)
feature_importance.sort_values(key=lambda x: abs(coefficients), ascending=False, inplace=True)

# print(feature_importance)
# print(accuracy)
