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
import os
import re
import ast

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./dataset_tomato.xlsx",index_col = 0,)
data

# %%
Y = data.iloc[:,0:5]
Y

# %%
y_name = 'Overall liking'
random_seed = 1
exec("custom_select = pd.read_excel(\"./Table/{}/Feature_select_custom_{}.xlsx\",index_col = 0,)".format(y_name,random_seed))
custom_select


# %%
custom_select = custom_select.sort_values(by=['5-fold CV'], ascending=False)
custom_select

# %%
min_index = custom_select.iloc[0:5,2].idxmin() # find the index with minimum custom number in the top five high-performanced models
min_index

# %%
selected_string = custom_select.loc[min_index,'Custom selected features']
selected_string

# %%
matches = re.findall(r"'([^']*)'", selected_string)
selected_features = list(matches)
selected_features

# %% Select important features for each model
for y_name in Y.columns[0:5]:
    features_result = pd.DataFrame(columns=custom_select.columns,index=range(1,11,1))
    for random_seed in range(1,11,1):
        exec("custom_select = pd.read_excel(\"./Table/{}/Feature_select_custom_{}.xlsx\",index_col = 0,)".format(y_name,random_seed))
        #custom_select = custom_select.drop(custom_select[custom_select['Custom number']==68].index)
        custom_select = custom_select.sort_values(by=['5-fold CV'], ascending=False)
        min_index = custom_select.iloc[0:5,2].idxmin() # find the index with minimum custom number in the top five high-performanced models
        selected_string = custom_select.loc[min_index,'Custom selected features']
        matches = re.findall(r"'([^']*)'", selected_string)
        selected_features = list(matches)
        features_result.loc[random_seed,:] = custom_select.loc[min_index,:]
        features_result.loc[random_seed,'Custom selected features'] = selected_features
    features_result.to_excel('./Table/Selected features/Each_selected_{}.xlsx'.format(y_name))


# %% select features for all models
final_selected_features = pd.DataFrame(columns=['Number','Selected features'],index=Y.columns[0:5])
for y_name in Y.columns[0:5]:
    exec("each_selected = pd.read_excel(\"./Table/Selected features/Each_selected_{}.xlsx\",index_col = 0,)".format(y_name))
    feature_list = []
    for i in range(1,11,1):
        temp_list = each_selected.loc[i,'Custom selected features']
        feature_list[len(feature_list):] = ast.literal_eval(temp_list)
    feature_list_df = pd.DataFrame(feature_list,columns=['Feature'])

    frequency = feature_list_df['Feature'].value_counts()
    feature_no_less_3 = frequency[frequency >= 3].index.tolist()
    final_selected_features.loc[y_name,'Selected features'] = feature_no_less_3
    final_selected_features.loc[y_name,'Number'] = len(feature_no_less_3)
final_selected_features.to_excel('./Table/Selected features/Final_selected_features.xlsx')
final_selected_features

# %%
# %%
