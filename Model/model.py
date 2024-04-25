from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd


RANDOM_STATE = 42
DIR = 'Model/'

class Model:
    def __init__(self, data):
        self.data = data
        self.model = LogisticRegression(random_state=RANDOM_STATE)
        self.scaler = StandardScaler()

    def train(self, target='Target'):
        X = self.data.drop(target, axis=1)
        y = self.data[target]

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=RANDOM_STATE)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save_model(self, model_filename='model.pkl', scaler_filename='scaler.pkl'):
        joblib.dump(self.model, DIR + model_filename)
        joblib.dump(self.scaler, DIR + scaler_filename)



data = pd.read_csv(DIR + 'data.csv', sep=';')
model = Model(data)
print(model.train(target='Target'))


model.save_model()




