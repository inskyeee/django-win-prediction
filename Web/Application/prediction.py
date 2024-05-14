import joblib
import json
import pandas as pd


with open('../Model/heroes.json', 'r') as file:
    heroes = pd.json_normalize(json.load(file)['heroes'])
    hero_names = heroes.localized_name.to_list()

    
HERO_DATA = {hero_names[i]: 0 for i in range(0, len(hero_names))}


loaded_model = joblib.load('../Model/model.pkl')
loaded_scaler = joblib.load('../Model/scaler.pkl')


def process_game(game):
    hero_data = HERO_DATA.copy()
    team1_hero1 = game["team1_hero1"]
    team1_hero2 = game["team1_hero2"]
    team1_hero3 = game["team1_hero3"]
    team1_hero4 = game["team1_hero4"]
    team1_hero5 = game["team1_hero5"]
    
    team2_hero1 = game["team2_hero1"]
    team2_hero2 = game["team2_hero2"]
    team2_hero3 = game["team2_hero3"]
    team2_hero4 = game["team2_hero4"]
    team2_hero5 = game["team2_hero5"]

    hero_data[team1_hero1] = 1
    hero_data[team1_hero2] = 1
    hero_data[team1_hero3] = 1
    hero_data[team1_hero4] = 1
    hero_data[team1_hero5] = 1

    hero_data[team2_hero1] = -1
    hero_data[team2_hero2] = -1
    hero_data[team2_hero3] = -1
    hero_data[team2_hero4] = -1
    hero_data[team2_hero5] = -1

    hero_data = pd.DataFrame(hero_data, index=[0])
    hero_data_scaled = loaded_scaler.transform(hero_data)
    prediction_proba = loaded_model.predict_proba(hero_data_scaled)
    prediction = loaded_model.predict(hero_data_scaled)
    return prediction_proba[0].tolist()