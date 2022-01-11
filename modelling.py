
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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
                Random Forest Model creation & Evaluation


'''

# Spin up a Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators = 1600, min_samples_split = 5, min_samples_leaf = 1, max_features = 'auto',max_depth = 10, bootstrap = True)

model_rf.fit(x_train, y_train)

yhat_train = list(model_rf.predict(x_train))

training_error = mean_squared_error(y_train, yhat_train)


results = pd.DataFrame(x_train)



yhat_test = list(model_rf.predict(x_test))


test_error = mean_squared_error(y_test, yhat_test)



test_results = pd.DataFrame(x_test)

test_results['goal_pred'] = yhat_test

test_results['goals'] = list(y_test)


'''
                K-Nearest Neighbours Hyperparameter Tuning


'''

#List Hyperparameters that we want to tune.
leaf_size = list(range(1,50))
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn = KNeighborsClassifier()
#Use GridSearch
clf = RandomizedSearchCV(estimator = knn, param_distributions = hyperparameters,n_iter = 100, cv=10, random_state = 42)
#Fit the model
best_model = clf.fit(x,y)
#Print The value of best Hyperparameters
print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])



'''
            K-Nearest Neighbours Model Creation & Evaluation

'''

model_knn = KNeighborsClassifier(leaf_size = 31, p =2, n_neighbors = 26)

model_knn.fit(x_train, y_train)

knn_yhat_train = list(model_knn.predict(x_train))

knn_training_error = mean_squared_error(knn_yhat_train, y_train)


knn_yhat_test = list(model_knn.predict(x_test))

knn_test_error = mean_squared_error(knn_yhat_test, y_test)

test_results['knn_goal_pred'] = knn_yhat_test


