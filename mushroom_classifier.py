# https://archive.ics.uci.edu/dataset/73/mushroom
# https://github.com/uci-ml-repo/ucimlrepo

# pip install -r /path/to/requirements.txt

import pickle
from ucimlrepo import fetch_ucirepo     
import pandas as pd              
from sklearn.tree import DecisionTreeClassifier     
from sklearn.model_selection import train_test_split       

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 

# data (as pandas dataframes) 
X = mushroom.data.features      # X (uppercase) is a feature matrix
y = mushroom.data.targets       # y (lowercase) is a target vector
  
# metadata 
for key in mushroom.metadata: 
    if key != 'additional_info':
        print(key, ":", mushroom.metadata[key])
    else:   # i.e. key == 'additional_info'
        for k in mushroom.metadata[key]:
            print(k, ":", mushroom.metadata[key][k])

# variable information 
print()
print(mushroom.variables) 



# Part 1 ------------------------------------------------------------------------------- #

# TODO: 'Clean' The Data

'''
Stalk-root feature (11) has missing values (?).
We'll need to handle this before we can use it in our model.
'''

# === Data Cleaning === #
### *First Drop the feature that has missing values



# ===================== #

print(X.head())

# === Convert To Numerical Values === #
### *Use pandas get_dummies function to convert the categorical features to numerical values


# =================================== #

print(X.head())

# Part 2 ------------------------------------------------------------------------------- #

score = 0

# Split the data into training and testing sets
# Seed for reproducibility
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# TODO: Create a Decision Tree Classifier



# TODO: Train and test the model



print()
print(f'Model Score: {score}')       # Should be pretty close to 1.0

# TODO: Pickle the model
# *(just uncomment the code below)*

'''
# dtc = Decision Tree Classifier :)
model_path = 'mushroom_dtc'
with open(model_path, 'wb') as file:
    print('Saving model...')
    pickle.dump(model, file)
    print('Model saved!')
'''


# Part 3 ------------------------------------------------------------------------------- #

# TODO: Load the model

# New features
# We'll use the lactarius_indigo mushroom (a milk mushroom) as an example https://en.wikipedia.org/wiki/Lactarius_indigo
# This mushroom is edible

new_features = {
    'cap-shape':                    ['x'],
    'cap-surface':                  ['s'],
    'cap-color':                    ['n'],
    'bruises':                      ['t'],
    'odor':                         ['n'],
    'gill-attachment':              ['f'],
    'gill-spacing':                 ['c'],
    'gill-size':                    ['n'],
    'gill-color':                   ['k'],
    'stalk-shape':                  ['e'],
    'stalk-surface-above-ring':     ['s'],
    'stalk-surface-below-ring':     ['s'],
    'stalk-color-above-ring':       ['w'],
    'stalk-color-below-ring':       ['w'],
    'veil-type':                    ['p'],
    'veil-color':                   ['w'],
    'ring-number':                  ['o'],
    'ring-type':                    ['p'],
    'spore-print-color':            ['k'],
    'population':                   ['s'],
    'habitat':                      ['d']
}

# TODO: Predict the class of the new features

# === Convert The New Features === #
X = mushroom.data.features
X = X.drop(columns='stalk-root')
# We'll append the new features to the dataset
features_df = pd.DataFrame(new_features)
X_new = pd.concat([X, features_df], ignore_index=True)
X_new = pd.get_dummies(X_new)
new_features = X_new.iloc[-1]

# ================================ #

# *Predict the class of the new features

