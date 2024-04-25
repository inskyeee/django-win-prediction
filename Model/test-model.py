import pandas as pd
from sklearn.metrics import accuracy_score
import joblib


loaded_model = joblib.load('Model/model.pkl')
loaded_scaler = joblib.load('Model/scaler.pkl')

data = pd.read_csv('Model/data.csv')
data_scaled = loaded_scaler.transform(data.drop('Target', axis=1))

predictions = loaded_model.predict(data_scaled)


print(accuracy_score(data['Target'], predictions))