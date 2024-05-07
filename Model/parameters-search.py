from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42
DIR = 'Model/'

train_data = pd.read_csv(DIR + 'dota2Train.csv')
train_data.columns = ['Target'] + ['Game mode'] + ['Game type'] + [f'Feature_{i}' for i in range(1, train_data.shape[1] - 2)]

test_data = pd.read_csv(DIR + 'dota2Test.csv')
test_data.columns = ['Target'] + ['Game mode'] + ['Game type'] + [f'Feature_{i}' for i in range(1, test_data.shape[1] - 2)]


# logreg = LogisticRegression(random_state=RANDOM_STATE)
# params_grid = {'C': [0.1, 1, 10, 100, 1000],
#                 'penalty': ['l1', 'l2'],
#                 'solver': ['liblinear', 'saga']}

# clf = RandomizedSearchCV(logreg, params_grid, n_iter=10, cv=5, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')
# clf.fit(train_data.drop('Target', axis=1), train_data['Target'])
# results = pd.DataFrame(clf.cv_results_)

svm = SVC(random_state=RANDOM_STATE)
params_grid = {'C': [0.1, 1, 10, 100, 1000],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4],
                'gamma': ['scale', 'auto']}
clf = RandomizedSearchCV(svm, params_grid, n_iter=10, cv=5, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')
clf.fit(train_data.drop('Target', axis=1), train_data['Target'])
results = pd.DataFrame(clf.cv_results_)
print(results[['params', 'mean_test_score', 'rank_test_score']])


# RandomForest = RandomForestClassifier(random_state=RANDOM_STATE)
# params_grid = {'n_estimators': [100, 200, 300, 400, 500],
#                 'max_depth': [10, 20, 30, 40, 50],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'bootstrap': [True, False]}
# clf = RandomizedSearchCV(RandomForest, params_grid, n_iter=10, cv=5, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')
# clf.fit(train_data.drop('Target', axis=1), train_data['Target'])
# results = pd.DataFrame(clf.cv_results_)
# print(results[['params', 'mean_test_score', 'rank_test_score']])

# knn = KNeighborsClassifier()
# params_grid = {'n_neighbors': [5, 10, 15, 20, 25],
#                 'weights': ['uniform', 'distance'],
#                 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#                 'p': [1, 2]}
# clf = RandomizedSearchCV(knn, params_grid, n_iter=10, cv=5, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')
# clf.fit(train_data.drop('Target', axis=1), train_data['Target'])
# results = pd.DataFrame(clf.cv_results_)
# print(results[['params', 'mean_test_score', 'rank_test_score']])

# dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
# params_grid = {'max_depth': [10, 20, 30, 40, 50],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]}
# clf = RandomizedSearchCV(dt, params_grid, n_iter=10, cv=5, n_jobs=-1, random_state=RANDOM_STATE, scoring='accuracy')
# clf.fit(train_data.drop('Target', axis=1), train_data['Target'])
# results = pd.DataFrame(clf.cv_results_)
# print(results['params', 'mean_test_score', 'rank_test_score'])

