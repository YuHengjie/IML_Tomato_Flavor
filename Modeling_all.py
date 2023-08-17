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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imodels import RuleFitClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier

from xgboost import XGBClassifier
import lightgbm as lgb
import re
import ast
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
feature_selected = pd.read_excel("./Table/Selected features/Final_selected_features.xlsx",index_col = 0,)
feature_selected

# %%
Y = data.iloc[:,0:5]
Y

# %%
X = data.iloc[:,5:] # 'TEM size (nm)'
X

# %%
y_name = data.columns[0]
y = data.loc[:,y_name]
y

# %%
random_seed = 1
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed) 
for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

feature_name = feature_selected.loc[y_name,'Selected features']
X_test = X_test.loc[:,ast.literal_eval(feature_name)]
X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]
X_cv

# %%
param_grid = {
    # 'n_estimators': [100, 200, 400, 800],
    # 'max_depth': [3, 4, 5],
    # 'max_features': np.linspace(2,X_cv.shape[1],4).astype(int),
    # 'min_samples_split': [2, 4, 6, 8],
    # 'min_samples_leaf': [1, 2, 4],
}

rfc = RandomForestClassifier(random_state = 42, n_jobs = -1,bootstrap = True,
                              n_estimators = 100,
                              max_depth = 5,
                              max_features = 'log2',
                              min_samples_split = 2,
                              min_samples_leaf = 1,
                             )

gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
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
# Evaluate the random forest classifier on the test set
accuracy = rfc_best.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)
# Evaluate the random forest classifier on the training set
accuracy = rfc_best.score(X_cv, y_cv)
print("Accuracy on the training set:", accuracy)

# %% 1、Random Forest
test_ratio = 0.25

best_parameters = []
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: Random Forest ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        param_grid = {
             'n_estimators': [100, 200, 400,],
             'max_depth': [3, 4, 5, 6],
             'max_features': np.linspace(2,min(10, X_cv.shape[1]),5).astype(int),
             'min_samples_split': [2, 4, 7, 10, 15],
             'min_samples_leaf': [1, 2, 4, 6, 8],
             "max_leaf_nodes":[10, 13, 16, 19, 22],
        }
        rfc = RandomForestClassifier(random_state=42, n_jobs=-1,bootstrap=True)
        gs = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(X_cv, y_cv)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        rfc_best = gs.best_estimator_

        y_pred_cv = rfc_best.predict(X_cv)
        y_proba_cv = rfc_best.predict_proba(X_cv)[:, 1]

        y_pred = rfc_best.predict(X_test)
        y_proba = rfc_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(rfc_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(rfc_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(rfc_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/RF/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/RF/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/RF/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)

# %%

# %% 2、XGBoost
test_ratio = 0.25

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: XGBoost ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        param_grid = {
            'n_estimators': [100, 200, 400,],
            'max_depth': [3, 4, 5, 6,],
            'min_child_weight': [1, 2, 5, 8,],
            'gamma': [0, 0.5, 1, 5,],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'reg_lambda':[0.1, 1, 10, 50, 100]

        }
        xgb = XGBClassifier(random_state=42, n_jobs=-1,)
        gs = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(X_cv, y_cv,)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        xgb_best = gs.best_estimator_

        y_pred_cv = xgb_best.predict(X_cv)
        y_proba_cv = xgb_best.predict_proba(X_cv)[:, 1]

        y_pred = xgb_best.predict(X_test)
        y_proba = xgb_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(xgb_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(xgb_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(xgb_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/XGBoost/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/XGBoost/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/XGBoost/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)


# %% 3. LGBMClassifier
test_ratio = 0.25

def clean_feature_names(feature_names):
    cleaned_names = []
    for name in feature_names:
        cleaned_name = re.sub(r'\W+', '_', name)  # Replace non-alphanumeric characters with underscores
        cleaned_names.append(cleaned_name)
    return cleaned_names

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: LightGBM ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]

        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        name_mapping = dict(zip(X_cv.columns, clean_feature_names(X_cv.columns)))
        X_cv.rename(columns=name_mapping, inplace=True)
        X_test.rename(columns=name_mapping, inplace=True)

        param_grid = {
            'boosting_type': ['gbdt','dart'],
            "num_iterations":[100,200,400,],
            "max_bin":  [8,16,24,32],
            "min_child_samples":[4,8,12,20,],
            "min_child_weight": [0.001,0.01,0.1,1],
            "max_depth":[3,4,5,6,],
            "num_leaves": [4,8,12,16,20],
            'learning_rate':[0.01,0.02,0.05,0.1,],
        }

        lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1,)
        gs = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5,scoring='accuracy', n_jobs=-1)
        gs.fit(X_cv, y_cv)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        lgbm_best = gs.best_estimator_

        y_pred_cv = lgbm_best.predict(X_cv)
        y_proba_cv = lgbm_best.predict_proba(X_cv)[:, 1]

        y_pred = lgbm_best.predict(X_test)
        y_proba = lgbm_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(lgbm_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(lgbm_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(lgbm_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/LightGBM/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/LightGBM/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/LightGBM/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)




# %% 4. DT
test_ratio = 0.25

best_parameters = []
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: DT ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        param_grid = {
             'max_depth': range(3,7,2),
             'splitter': ['best','random'],
             'max_features': np.linspace(2,min(14, X_cv.shape[1]),5).astype(int),
             'min_samples_split': range(2,15,2),
             'min_samples_leaf': range(1,21,3),
             'max_leaf_nodes': range(6,21,2),
             'criterion': ['gini', 'entropy', 'log_loss'],
             'ccp_alpha': np.arange(0,1,0.1),
        }
        dt = DecisionTreeClassifier(random_state=42, )
        gs = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(X_cv, y_cv)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        dt_best = gs.best_estimator_

        y_pred_cv = dt_best.predict(X_cv)
        y_proba_cv = dt_best.predict_proba(X_cv)[:, 1]

        y_pred = dt_best.predict(X_test)
        y_proba = dt_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(dt_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(dt_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(dt_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/DT/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/DT/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/DT/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)


# %% 5. MLP
test_ratio = 0.25

best_parameters = []
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: MLP ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        param_grid = {
             'hidden_layer_sizes': [(10,), (20,), (40,),(70,), (100,),(150,)],
             'learning_rate': ['constant', 'invscaling', 'adaptive'],
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 0.5,1],
             'max_iter': [100, 200, 400, 600],
             'batch_size': [8, 16, 32, 64, 128,],

        }
        mlp = MLPClassifier(random_state = 42, early_stopping= True,)
        gs = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5,scoring='accuracy',n_jobs=-1)
        gs.fit(X_cv, y_cv)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        mlp_best = gs.best_estimator_

        y_pred_cv = mlp_best.predict(X_cv)
        y_proba_cv = mlp_best.predict_proba(X_cv)[:, 1]

        y_pred = mlp_best.predict(X_test)
        y_proba = mlp_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(mlp_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(mlp_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(mlp_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/MLP/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/MLP/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/MLP/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)



# %% 6. RuleFit
test_ratio = 0.25

for y_name in Y.columns[0:5]:
    best_parameters = []
    df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))

    y = Y.loc[:,y_name]
    
    for random_seed in range(1,11,1):
        print('Processing: RuleFit ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]

        feature_name = feature_selected.loc[y_name,'Selected features']
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]

        param_grid = {
                 "n_estimators":[100,200,400,800],
                 "tree_generator":[None,RandomForestRegressor(),GradientBoostingRegressor(),GradientBoostingClassifier()],
                 "tree_size":[3,4,5,6,7],
                 "max_rules":[15,20,25,30,35,40,45,50],
        }

        rulefit = RuleFitClassifier(random_state=42, include_linear=False)
        gs = GridSearchCV(estimator=rulefit, param_grid=param_grid, cv=5,scoring='accuracy', n_jobs=-1)
        gs.fit(X_cv, y_cv)

        best_parameters.append(["Best parameters for "+str(random_seed)+": ",gs.best_params_,])

        rulefit_best = gs.best_estimator_

        y_pred_cv = rulefit_best.predict(X_cv)
        y_proba_cv = rulefit_best.predict_proba(X_cv)[:, 1]

        y_pred = rulefit_best.predict(X_test)
        y_proba = rulefit_best.predict_proba(X_test)[:, 1]
        
        df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(rulefit_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                    'F1_CV':sum(cross_val_score(rulefit_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                    'Accuracy_CV':sum(cross_val_score(rulefit_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                    'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                    'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                    'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                    'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                    'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                    'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
        
        df_result.to_excel('./Model/RuleFit/{}/GridSearch_performance_{}.xlsx'.format(y_name,random_seed))
        X_test['Observed RMC'] = y_test
        X_test['Predicted RMC'] = y_pred
        X_test.to_excel('./Model/RuleFit/{}/Predict_observe_{}.xlsx'.format(y_name,random_seed))

        list_str = '\n'.join(str(item) for item in best_parameters)
        # Write string to file
        with open('./Model/RuleFit/{}/GridSearch_parameters_{}.txt'.format(y_name,random_seed), 'w') as file:
            file.write(list_str)

# %%
