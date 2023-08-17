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
from sklearn.inspection import permutation_importance

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
y_name = ['Overall liking', 'Sweetness', 'Sourness', 'Umami', 'Flavor intensity']
y_name = 'Sweetness'
y = data.loc[:,y_name]
y

# %%
feature_selected = pd.read_excel("./Table/Selected features/Final_selected_features.xlsx",index_col = 0,)
feature_selected

# %%
random_seed = 1

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


        RF_importance = model_best.feature_importances_

        result = permutation_importance(model_best, X_cv, y_cv, scoring='accuracy', 
                                    n_repeats=10, random_state=0, n_jobs=-1)
        Permutation_importance = result.importances_mean

        explainer = shap.TreeExplainer(model=model_best, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
        shap_values = explainer.shap_values(X_cv)[1]
        global_shap_values = np.abs(shap_values).mean(0)

        # Calculate the relative importance, the maximum is 1
        RF_importance_relative = RF_importance/max(RF_importance)
        Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
        shap_values__relative = global_shap_values/max(global_shap_values)

        # Sort by the sum of relative importance
        importance_sum = RF_importance_relative+Permutation_importance_relative+shap_values__relative
        sorted_idx_sum = importance_sum.argsort()
        sorted_features = X_cv.columns[sorted_idx_sum][::-1]

        importance_df.loc[:, random_seed] = importance_sum/3

    importance_df['Mean'] = importance_df.mean(axis=1)
    importance_df['SD'] = importance_df.std(axis=1)

    importance_df.to_excel('./Table/Importance/Relative_Importance_{}.xlsx'.format(y_name))


# %%
feature_all = []
for y_name in Y.columns[0:5]:
    importance_df = pd.read_excel("./Table/Importance/Relative_Importance_{}.xlsx".format(y_name),index_col = 0,)
    feature_all.extend(importance_df.index)
feature_all = np.array(feature_all)
feature_all = np.unique(feature_all)
feature_all

# %%
importance_all = pd.DataFrame(index=Y.columns[0:5],columns=feature_all)
for y_name in Y.columns[0:5]:
    importance_df = pd.read_excel("./Table/Importance/Relative_Importance_{}.xlsx".format(y_name),index_col = 0,)
    for feature in importance_df.index:
        importance_all.loc[y_name,feature] = importance_df.loc[feature,'Mean']
importance_all = importance_all.astype('float')
importance_all.to_excel('importance_all.xlsx')
importance_all

# %%
fig, ax= plt.subplots(figsize = (18, 4))
plt.style.use('default')
cmap = sns.diverging_palette(220, 20, as_cmap=True)
cmap.set_bad('white')  # Set NaN values to white color

h=sns.heatmap(importance_all, cmap=cmap,  square=True, center=0.45,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8},
            mask=importance_all.isnull())
bottom, top = ax.get_ylim()
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Importance_all.jpg",dpi=600,bbox_inches='tight')

# %%
importance_all.fillna(0, inplace=True)
mean_row = pd.DataFrame(importance_all.mean(axis=0).values.reshape(1, -1), columns=importance_all.columns, index=['Mean'])
importance_all = importance_all.append(mean_row, ignore_index=False)
importance_all.replace(0, np.nan, inplace=True)
importance_all

# %%

fig, ax= plt.subplots(figsize = (18, 4))
plt.style.use('default')
cmap = sns.diverging_palette(220, 20, as_cmap=True)
cmap.set_bad('white')  # Set NaN values to white color

h=sns.heatmap(importance_all, cmap=cmap,  square=True, center=0.4,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8},
            mask=importance_all.isnull())
bottom, top = ax.get_ylim()
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Importance_all_mean.jpg",dpi=600,bbox_inches='tight')
# %%
