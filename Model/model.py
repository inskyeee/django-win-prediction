from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import json
import joblib
import pandas as pd


RANDOM_STATE = 42
DIR = 'Model/'

class Model:    
    def __init__(self, train_data, test_data, model_type='logreg'):
        self.train_data = train_data
        self.test_data = test_data
        self.scaler = StandardScaler()
        self.feature_names = train_data.columns.drop('Target')
        self.model_type = model_type

        if self.model_type == 'logreg':
            self.model = LogisticRegression(random_state=RANDOM_STATE, C=0.1, solver='liblinear', penalty='l1')
        elif self.model_type == 'svm':
            self.model = SVC(random_state=RANDOM_STATE)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.model_type == 'knn':
            self.model = KNeighborsClassifier()
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=RANDOM_STATE)



    def split_data(self, train_data, test_data, target='Target'):
        X_train = self.train_data.drop(target, axis=1)
        X_test = self.test_data.drop(target, axis=1)
        y_train = self.train_data[target]
        y_test = self.test_data[target]
        
        if self.model_type == 'logreg':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    def train(self, target='Target'):
        X_train, X_test, y_train, y_test = self.split_data(self.train_data, self.test_data, target)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        if self.model_type == 'logreg':
            self.coefficients = self.model.coef_[0]

        return accuracy_score(y_test, y_pred)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, model_filename='model.pkl', scaler_filename='scaler.pkl'):
        joblib.dump(self.model, DIR + model_filename)
        joblib.dump(self.scaler, DIR + scaler_filename)



with open('Model/heroes.json', 'r') as file:
    heroes = pd.json_normalize(json.load(file)['heroes'])
    hero_names = heroes.localized_name.to_list()

train_data = pd.read_csv(DIR + 'dota2Train.csv')
train_data.columns = ['Target'] + [i for i in hero_names]

test_data = pd.read_csv(DIR + 'dota2Test.csv')
test_data.columns = ['Target'] + [i for i in hero_names]

model = Model(train_data, test_data)
accuracy = model.train(target='Target')
print("Training accuracy:", model.train(target='Target'))


model.save_model()




