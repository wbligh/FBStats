
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



fb_data = pd.read_csv('training_data.csv')

fb_data['year'] = fb_data['short_date'].str[0:4]

'''
                Data Pre-processing


'''

# First we're going to simply remove any rows with nan values
fb_data_clean = fb_data.dropna()



# convert the home/away column to binary
games = {'h': 1,'a': 0}

fb_data_clean.h_a = [games[item] for item in fb_data_clean.h_a]

numeric_cols = fb_data_clean.select_dtypes(include=['float64','int64']).columns.to_list()


# Review distributions for the numeric data

for i, col in enumerate(numeric_cols):
    plt.figure(i)
    sns.distplot(fb_data_clean[col])

# Pertinent numeric  distrubutions are approximately Gaussian 
    

data = fb_data_clean[numeric_cols].drop(columns = ['Match_ID','match_rank'])

# Split our x and y data
x = data.drop(columns = ['goals'])
y = data.goals

# Make our train test splits on the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# Scale our data

sc = StandardScaler() 

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



'''
            Random Forest Hyperparameter Tuning


'''



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)


param_grid = {'n_estimators': [1550,1570,1590,1600,1620,1640,1660],
               'max_features': ['auto'],
               'max_depth': [6,8,10,12],
               'min_samples_split': [4,5,6],
               'min_samples_leaf': [1,2],
               'bootstrap': [True]}


rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

rf_grid.fit(x_train, y_train)


# =============================================================================
#   Best params given by:
# 
# {'n_estimators': 1600,
#  'min_samples_split': 5,
#  'min_samples_leaf': 1,
#  'max_features': 'auto',
#  'max_depth': 10,
#  'bootstrap': True}
# =============================================================================


'''
                Model creation & Evaluation


'''

# Spin up a Random Forest Classifier
model = RandomForestClassifier(n_estimators = 1600, min_samples_split = 5, min_samples_leaf = 1, max_features = 'auto',max_depth = 10, bootstrap = True)

model.fit(x_train, y_train)

yhat_train = list(model.predict(x_train))

training_error = mean_squared_error(y_train, yhat_train)


results = pd.DataFrame(x_train)



yhat_test = list(model.predict(x_test))

test_results = pd.DataFrame(x_test)

test_results['goal_pred'] = yhat_test

test_results['goals'] = list(y_test)


test_error = mean_squared_error(y_test, yhat_test)

