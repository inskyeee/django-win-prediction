import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import json
pd.set_option('display.max_columns', None)

loaded_model = joblib.load('Model/model.pkl')
loaded_scaler = joblib.load('Model/scaler.pkl')


with open('Model/heroes.json', 'r') as file:
    heroes = pd.json_normalize(json.load(file)['heroes'])
    hero_names = heroes.localized_name.to_list()
    print(hero_names)



data = pd.read_csv('Model/dota2Test.csv')

data.columns = ['Target'] + [i for i in hero_names]
data_scaled = loaded_scaler.transform(data.drop('Target', axis=1))

predictions = loaded_model.predict(data_scaled)
accuracy = accuracy_score(data['Target'], predictions)

coefficients = loaded_model.coef_[0]
features = loaded_scaler.feature_names_in_

feature_importance = pd.Series(coefficients, index=features)
feature_importance.sort_values(key=lambda x: abs(coefficients), ascending=False, inplace=True)



