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

# %%  Sweetness
y_name = 'Sweetness' #  ['Overall liking', 'Sweetness', 'Sourness', 'Umami', 'Flavor intensity']
y = data.loc[:,y_name]
y

# %%
feature_selected = pd.read_excel("./Table/Selected features/Final_selected_features.xlsx",index_col = 0,)
feature_selected

# %%
feature_name = feature_selected.loc[y_name,'Selected features']
feature_name_list = ast.literal_eval(feature_name)
feature_name_list

# %%
X = data.loc[:,feature_name_list]
X

# %%
cmap = sns.diverging_palette(220, 20, as_cmap=True)

X_corr = X.copy()
corr = X_corr.corr()

fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')

h=sns.heatmap(corr, cmap=cmap,  square=True, center=0.5,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':8})
bottom, top = ax.get_ylim()

cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/{}/X_corr.jpg".format(y_name),dpi=600,bbox_inches='tight')

# %%
shap_values_all = np.load('./SHAP/shap_values_{}.npy'.format(y_name))
shap_interaction_values_all = np.load('./SHAP/shap_interaction_values_{}.npy'.format(y_name))
X_test_all = pd.read_excel('./SHAP/X_test_{}.xlsx'.format(y_name),index_col = 0,)
print(shap_interaction_values_all.shape, shap_values_all.shape, X_test_all.shape,)

# %%
print(shap_values_all.min(),shap_values_all.max())

# %%
for feature in feature_name_list:
    index = feature_name_list.index(feature)
    global_main_effects = shap_values_all[:,index]
    fig, ax = plt.subplots(figsize=(2.2, 1.1))
    plt.xlabel(feature)
    plt.ylabel('SHAP value')
    plt.ylim(-0.42,0.31)
    plt.yticks([-0.3,0,0.3])
    plt.scatter(X_test_all[feature], global_main_effects, marker='o',
                s=30,c='#EA8D6C',linewidth=0.2,edgecolors='#FFFFFF')
    fig.savefig("./Image/{}/SHAP_{}.jpg".format(y_name,feature),dpi=600,bbox_inches='tight')


# %% SHAP interaction values
fig, ax= plt.subplots(figsize = (12, 12))
plt.style.use('default')
tmp = np.abs(shap_interaction_values_all).sum(0)/10
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = [feature_name_list[item] for item in inds]
df_temp2.index = [feature_name_list[item] for item in inds]

h=sns.heatmap(df_temp2, cmap='flare', square=True, center=0.3,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
bottom, top = ax.get_ylim()

cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=15, rotation_mode='anchor')

df_temp2.to_excel('./Table/Importance/Shap_Inter_values_{}.xlsx'.format(y_name))

fig.savefig("./Image/{}/SHAP_Interactions_{}.jpg".format(y_name, y_name),dpi=600,bbox_inches='tight')


# %% 1 interactions
feature_1 = 'fructose'
feature_2 = '2,5-dimethyl-4-hydroxy-3(2H)-furanone'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(3, 2))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

sm = plt.cm.ScalarMappable(cmap="flare",)
sm.set_array([data_plot[feature_2].min(),data_plot[feature_2].max()])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/{}/SHAP_inter_{}_{}.jpg".format(y_name, feature_1,feature_2),dpi=600,bbox_inches='tight')

# %% 2 interactions
feature_1 = 'fructose'
feature_2 = '2-isobutylthiazole'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(3, 2))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

sm = plt.cm.ScalarMappable(cmap="flare",)
sm.set_array([data_plot[feature_2].min(),data_plot[feature_2].max()])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/{}/SHAP_inter_{}_{}.jpg".format(y_name, feature_1,feature_2),dpi=600,bbox_inches='tight')

# %% 3 interactions
feature_1 = 'fructose'
feature_2 = 'benzyl cyanide'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(3, 2))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

sm = plt.cm.ScalarMappable(cmap="flare",)
sm.set_array([data_plot[feature_2].min(),data_plot[feature_2].max()])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/{}/SHAP_inter_{}_{}.jpg".format(y_name, feature_1,feature_2),dpi=600,bbox_inches='tight')

# %%
