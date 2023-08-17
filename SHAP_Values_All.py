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
from xgboost import XGBClassifier
import lightgbm as lgb
from imodels import RuleFitClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier

import shap
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
Y = data.iloc[:,0:5]
Y

# %%
X = data.iloc[:,5:]
X

# %%
feature_selected = pd.read_excel("./Table/Selected features/Final_selected_features.xlsx",index_col = 0,)
feature_selected

# %%
test_ratio = 0.25

for y_name in Y.columns[0:5]:
    y = data.loc[:,y_name]
    feature_name = feature_selected.loc[y_name,'Selected features']
    model_paras = pd.read_excel("./Parameters_RF_{}.xlsx".format(y_name),index_col = 0,)
    importance_df = pd.DataFrame(index=ast.literal_eval(feature_name),columns=range(1,11,1))

    for random_seed in range(1,11,1):
        print('Processing: ' + y_name + '_' + str(random_seed)+' ...')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]
        X_test = X_test.loc[:,ast.literal_eval(feature_name)]
        X_cv = X_cv.loc[:,ast.literal_eval(feature_name)]
        
        model_best = RandomForestClassifier(random_state=42, n_jobs=-1,bootstrap=True,
                                n_estimators = model_paras.loc[random_seed,'n_estimators'],
                                max_depth = model_paras.loc[random_seed,'max_depth'],
                                max_features = model_paras.loc[random_seed,'max_features'],
                                min_samples_split = model_paras.loc[random_seed,'min_samples_split'],
                                min_samples_leaf = model_paras.loc[random_seed,'min_samples_leaf'],
                                max_leaf_nodes = model_paras.loc[random_seed,'max_leaf_nodes'],
                                )

        model_best.fit(X_cv, y_cv,)

        y_pred_cv = model_best.predict(X_cv)
        y_proba_cv = model_best.predict_proba(X_cv)[:, 1]

        y_pred = model_best.predict(X_test)
        y_proba = model_best.predict_proba(X_test)[:, 1]

        # test the model is same with grid search
        print('Train Accuracy: %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
        print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))

        if random_seed == 1:
            shap_interaction_values_all = shap.TreeExplainer(model_best).shap_interaction_values(X_test)
            shap_interaction_values_all = np.array(shap_interaction_values_all)[1,:,:,:]
            shap_values_all = shap.TreeExplainer(model_best).shap_values(X_test)[1]
            X_test_all = X_test
        else:     
            shap_interaction_values = shap.TreeExplainer(model_best).shap_interaction_values(X_test)
            shap_interaction_values = np.array(shap_interaction_values)[1,:,:,:]
            shap_interaction_values_all = np.concatenate((shap_interaction_values_all,shap_interaction_values),axis=0)
            shap_values = shap.TreeExplainer(model_best).shap_values(X_test)[1]
            shap_values_all = np.append(shap_values_all,shap_values,axis=0)
            X_test_all = pd.concat([X_test_all, X_test])

    np.save('./SHAP/shap_values_{}.npy'.format(y_name), shap_values_all)
    np.save('./SHAP/shap_interaction_values_{}.npy'.format(y_name), shap_interaction_values_all)
    X_test_all.to_excel('./SHAP/X_test_{}.xlsx'.format(y_name))
    print(shap_interaction_values_all.shape, shap_values_all.shape, X_test_all.shape,)

# %%
