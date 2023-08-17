# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import ast
from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./dataset_tomato.xlsx",index_col = 0,)
data

# %% 
y_name = 'Sourness' #  ['Overall liking', 'Sweetness', 'Sourness', 'Umami', 'Flavor intensity']
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
for random_seed in range(1,11,1):
    exec("model_{} = joblib.load(\"./PDP_ICE/model_{}_{}.pkl\")".format(random_seed, y_name, random_seed))

# %%
X_test_all = pd.read_excel('./PDP_ICE/X_test_{}.xlsx'.format(y_name),index_col = 0,)
print(X_test_all.shape,)

# %%  numerical features for all models
ratio_show_lines = 1

# %%
feature = feature_name_list[2]
for i in range(1,11,1):

    exec("model_best = model_{}".format(i))
    X_test = X_test_all.iloc[53*(i-1):53*i,:]

    pdp_NP_none_M = pdp.pdp_isolate(model=model_best,
                        dataset=X_test,
                        model_features=X_test.columns,
                        feature=feature,
                        percentile_range=(0, 100),
                        n_jobs=-1, num_grid_points = min(20, len(np.unique(X_test[feature]))))
    if i == 1:
        ICE_lines = pdp_NP_none_M.ice_lines
        PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
        PDP_lines.loc[len(PDP_lines)] = pdp_NP_none_M.pdp
    else:
        ICE_lines = pd.concat([ICE_lines, pdp_NP_none_M.ice_lines])
        ICE_lines = ICE_lines.reset_index(drop=True)
        PDP_lines_temp = pd.DataFrame(columns=pdp_NP_none_M.ice_lines.columns)
        PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_NP_none_M.pdp
        PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
        PDP_lines = PDP_lines.reset_index(drop=True)

# %%
ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1) # sort columns for plot
num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
ICE_lines_show = ICE_lines.loc[select_lines]
ICE_lines_show

# %%
pdp_min = ICE_lines_show.columns.min()
pdp_max = ICE_lines_show.columns.max()
threshold = (pdp_max-pdp_min)/20

# %%
PDP_lines_df_raw = pd.DataFrame(index=range(0,20,1),columns=['x', 'x_mean','y','num','y_mean',])

for i in range(0,20,1):
    index = np.where((ICE_lines_show.columns.values>=pdp_min+threshold*i) & (ICE_lines_show.columns.values<=pdp_min+threshold*(i+1)))[0]
    df_temp = ICE_lines_show.iloc[:,index]
    y_item = df_temp.values[~np.isnan(df_temp.values)]
    PDP_lines_df_raw.loc[i,'x'] = str(ICE_lines_show.columns.values[index])
    PDP_lines_df_raw.loc[i,'x_mean'] = ICE_lines_show.columns.values[index].mean()

    PDP_lines_df_raw.loc[i,'y'] = str(y_item)
    PDP_lines_df_raw.loc[i,'num'] = len(y_item)
    PDP_lines_df_raw.loc[i,'y_mean'] = y_item.mean()
PDP_lines_df_raw = PDP_lines_df_raw[~PDP_lines_df_raw['x'].str.contains('\[\]')]
PDP_lines_df_raw = PDP_lines_df_raw.reset_index(drop=True)
PDP_lines_df_raw

# %%
def group_rows(column):
    # Initialize variables
    groups = []
    current_sum = 0
    current_indices = []

    # Iterate through the column values
    for i, value in enumerate(column):
        current_sum += value
        current_indices.append(i)

        # Check if the current sum exceeds the number
        if current_sum > 53*3: 
            groups.append(current_indices.copy())
            current_sum = 0
            current_indices.clear()

    return groups

# %%
# Sample data
data = {'column': [477, 212, 159, 265, 371, 53, 53, 159, 53, 106, 212, 106]}
df = pd.DataFrame(data)

# Call the function with the column values
groups = group_rows(df['column'])

# Print the resulting groups
for group in groups:
    print(group)

# %%
groups = group_rows(PDP_lines_df_raw['num'])
groups

# %%
PDP_lines_df = pd.DataFrame(index=range(0,len(groups),1),columns=['x_mean','num','y_mean',])

for i,indices in enumerate(groups):
    temp_df = PDP_lines_df_raw.iloc[indices,:]
    if temp_df.shape[0] == 1:
        PDP_lines_df.loc[i,'x_mean'] = temp_df['x_mean'].values[0]
        PDP_lines_df.loc[i,'num'] = temp_df['num'].values[0]
        PDP_lines_df.loc[i,'y_mean'] = temp_df['y_mean'].values[0]
    else:
        num = temp_df['num'].sum()
        PDP_lines_df.loc[i,'num'] = num
        x_sum = 0
        y_sum = 0
        for j in temp_df.index:
            x_sum += temp_df.loc[j,'x_mean']*temp_df.loc[j,'num']
            y_sum += temp_df.loc[j,'y_mean']*temp_df.loc[j,'num']
        PDP_lines_df.loc[i,'x_mean'] = x_sum/num
        PDP_lines_df.loc[i,'y_mean'] = y_sum/num
PDP_lines_df


# %%
fig, ax = plt.subplots(figsize=(3, 1.85))

#plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
for i in range(len(ICE_lines_show)):
    pd_temp = ICE_lines_show.iloc[i,:].dropna()
    plt.plot(pd_temp.index,pd_temp.values,linewidth=0.2,color='#696969',zorder=-1)

plt.plot(PDP_lines_df['x_mean'], PDP_lines_df['y_mean'], marker='o',markersize=4,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')

plt.xlabel(feature)
plt.ylabel('Predicted RMC')
fig.savefig("./Image/{}/PDP_ICE_{}.jpg".format(y_name,feature),dpi=600,bbox_inches='tight')


# %%
for feature in feature_name_list:
    for i in range(1,11,1):

        exec("model_best = model_{}".format(i))
        X_test = X_test_all.iloc[53*(i-1):53*i,:]

        pdp_NP_none_M = pdp.pdp_isolate(model=model_best,
                            dataset=X_test,
                            model_features=X_test.columns,
                            feature=feature,
                            percentile_range=(0, 100),
                            n_jobs=-1, num_grid_points = min(20, len(np.unique(X_test[feature]))))
        if i == 1:
            ICE_lines = pdp_NP_none_M.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_NP_none_M.pdp
        else:
            ICE_lines = pd.concat([ICE_lines, pdp_NP_none_M.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_NP_none_M.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_NP_none_M.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1) # sort columns for plot
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    pdp_min = ICE_lines_show.columns.min()
    pdp_max = ICE_lines_show.columns.max()
    threshold = (pdp_max-pdp_min)/20

    PDP_lines_df_raw = pd.DataFrame(index=range(0,20,1),columns=['x', 'x_mean','y','num','y_mean',])

    for i in range(0,20,1):
        index = np.where((ICE_lines_show.columns.values>=pdp_min+threshold*i) & (ICE_lines_show.columns.values<=pdp_min+threshold*(i+1)))[0]
        df_temp = ICE_lines_show.iloc[:,index]
        y_item = df_temp.values[~np.isnan(df_temp.values)]
        PDP_lines_df_raw.loc[i,'x'] = str(ICE_lines_show.columns.values[index])
        PDP_lines_df_raw.loc[i,'x_mean'] = ICE_lines_show.columns.values[index].mean()

        PDP_lines_df_raw.loc[i,'y'] = str(y_item)
        PDP_lines_df_raw.loc[i,'num'] = len(y_item)
        PDP_lines_df_raw.loc[i,'y_mean'] = y_item.mean()
    PDP_lines_df_raw = PDP_lines_df_raw[~PDP_lines_df_raw['x'].str.contains('\[\]')]
    PDP_lines_df_raw = PDP_lines_df_raw.reset_index(drop=True)

    groups = group_rows(PDP_lines_df_raw['num'])

    PDP_lines_df = pd.DataFrame(index=range(0,len(groups),1),columns=['x_mean','num','y_mean',])

    for i,indices in enumerate(groups):
        temp_df = PDP_lines_df_raw.iloc[indices,:]
        if temp_df.shape[0] == 1:
            PDP_lines_df.loc[i,'x_mean'] = temp_df['x_mean'].values[0]
            PDP_lines_df.loc[i,'num'] = temp_df['num'].values[0]
            PDP_lines_df.loc[i,'y_mean'] = temp_df['y_mean'].values[0]
        else:
            num = temp_df['num'].sum()
            PDP_lines_df.loc[i,'num'] = num
            x_sum = 0
            y_sum = 0
            for j in temp_df.index:
                x_sum += temp_df.loc[j,'x_mean']*temp_df.loc[j,'num']
                y_sum += temp_df.loc[j,'y_mean']*temp_df.loc[j,'num']
            PDP_lines_df.loc[i,'x_mean'] = x_sum/num
            PDP_lines_df.loc[i,'y_mean'] = y_sum/num
    
    ICE_lines_show.to_csv("./PDP_ICE/ICE_{}_{}.zip".format(y_name,feature))
    PDP_lines_df.to_csv("./PDP_ICE/PDP_{}_{}.zip".format(y_name,feature))
    
    fig, ax = plt.subplots(figsize=(2, 1))

    #plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i,:].dropna()
        plt.plot(pd_temp.index,pd_temp.values,linewidth=0.1,color='#696969',zorder=-1)

    plt.plot(PDP_lines_df['x_mean'], PDP_lines_df['y_mean'], marker='o',markersize=4,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
    plt.yticks([0,0.5,1])
    plt.xlabel(feature)
    plt.ylabel('Predicted probability')
    fig.savefig("./Image/{}/PDP_ICE_{}.jpg".format(y_name,feature),dpi=600,bbox_inches='tight')

# %%
