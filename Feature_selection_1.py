# %% 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold 
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import warnings
warnings.filterwarnings("ignore")

# %% 
data = pd.read_excel("./dataset_tomato.xlsx",index_col = 0,)
data

# %%
Y = data.iloc[:,0:5]
Y

# %%
X = data.iloc[:,5:] # 'TEM size (nm)'
X

# %%
y = data.iloc[:,1]

# %%
random_seed = 1
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed) 
# the sixth dataset split was used for model interpretation
for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]
#np.save('X_cv_index_RDW.npy',X_cv.index)
#np.save('X_test_index_RDW.npy',X_test.index)
X_cv

# %%
param_grid = {
    #'n_estimators': [100, 200, 500, 800],
    #'max_depth': [None, 4, 7, 10],
    #'max_features': [None,'log2', 'sqrt'],
    #'min_samples_split': [2, 4, 7, 10],
    #'min_samples_leaf': [1, 2, 4, 7],
    #'bootstrap': [True, False],
}
rfc = RandomForestClassifier(random_state=42, n_jobs=-1,
                             )

gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,scoring='accuracy')
gs.fit(X_cv, y_cv)

print("Best hyperparameters:", gs.best_params_)
print("Best score:", gs.best_score_)

# %%
# Use the best hyperparameters to train a random forest classifier on the entire training set
rfc_best = RandomForestClassifier(random_state=42, **gs.best_params_)
rfc_best.fit(X_cv, y_cv)
# %%
rfc_best = gs.best_estimator_

# %%
# Evaluate the random forest classifier on the testing set
accuracy = rfc_best.score(X_test, y_test)
print("Accuracy on the testing set:", accuracy)

# %%
feature_importance = rfc_best.feature_importances_ 
feature_importance

# %%
model = SelectFromModel(rfc_best, prefit=True)
X_new = model.transform(X_cv)
X_new.shape

# %%
selected_features = X_cv.columns[model.get_support()]
selected_features

# %%
threshold = np.linspace(0,feature_importance.max(),30)
select_process = pd.DataFrame(columns=['5-fold CV','Threshold','Custom number','Custom selected features'],
                                index=range(1,len(threshold)+1,1))

index = 1
for i in threshold:
    custom_model = SelectFromModel(rfc_best, threshold=i,prefit=True)
    X_temp = custom_model.transform(X_cv)
    select_process.loc[index,'5-fold CV'] = cross_val_score(rfc,X_temp,y_cv,cv=5).mean()
    select_process.loc[index,'Threshold'] = i
    select_process.loc[index,'Custom number'] = X_temp.shape[1]
    select_process.loc[index,'Custom selected features'] = X_cv.columns[custom_model.get_support()]
    index += 1
select_process

# %%
test_ratio = 0.25

best_parameters = []

accuracy_perform = pd.DataFrame(columns=('5-fold CV','Training set','Test set'),index=range(1,11,1))
feature_selected = pd.DataFrame(columns=('Default number','Default selected features'),index=range(1,11,1))

for y_name in Y.columns[0:5]:
    y = Y.loc[:,y_name]
    for random_seed in range(1,11,1):
        print('Processing: ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]

        param_grid = {
            'n_estimators': [100, 200, 400, 600, 800],
            'max_depth': [None, 3, 5, 7],
            'max_features': [6,9,12,15,18],
            'min_samples_split': [2, 4, 7, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6, 10],
        }
        rfc = RandomForestClassifier(random_state=42, n_jobs=-1,bootstrap=True,)
        gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(X_cv, y_cv)

        accuracy_perform.loc[random_seed,'5-fold CV'] = gs.best_score_
        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])
        
        rfc_best = gs.best_estimator_
        accuracy_perform.loc[random_seed,'Training set'] = rfc_best.score(X_cv, y_cv)
        accuracy_perform.loc[random_seed,'Test set'] = rfc_best.score(X_test, y_test)
        accuracy_perform.to_excel('./Table/{}/Feature_select_RF_performance_{}.xlsx'.format(y_name,random_seed))

        feature_importance = rfc_best.feature_importances_ 
        model = SelectFromModel(rfc_best, prefit=True)
        X_new = model.transform(X_cv)

        feature_selected.loc[random_seed,'Default number'] = X_new.shape[1]
        feature_selected.loc[random_seed,'Default selected features'] = X_cv.columns[model.get_support()]
        feature_selected.to_excel('./Table/{}/Feature_select_default_{}.xlsx'.format(y_name,random_seed))

        threshold = np.linspace(0,feature_importance.max(),50)
        select_process = pd.DataFrame(columns=['5-fold CV','Threshold','Custom number','Custom selected features'],
                                        index=range(1,len(threshold)+1,1))

        index = 1
        for i in threshold:
            custom_model = SelectFromModel(rfc_best, threshold=i,prefit=True)
            X_temp = custom_model.transform(X_cv)
            select_process.loc[index,'5-fold CV'] = cross_val_score(rfc,X_temp,y_cv,cv=5).mean()
            select_process.loc[index,'Threshold'] = i
            select_process.loc[index,'Custom number'] = X_temp.shape[1]
            select_process.loc[index,'Custom selected features'] = X_cv.columns[custom_model.get_support()]
            index += 1
            
        select_process.to_excel('./Table/{}/Feature_select_custom_{}.xlsx'.format(y_name,random_seed))

        list_str = str(best_parameters)
        # Write string to file
        with open('./Model/Basic_RF/{}/Feature_select_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)

# %%
