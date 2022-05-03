import math
import os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LinearRegression as linear
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import graphviz
from pysr import PySRRegressor
from sklearn.preprocessing import MinMaxScaler

#%%

#get csv and create dataframe, determine initial size
os.chdir(r'C:\Users\patri\Documents\UofU\MSE7050')
PATH = os.getcwd()
path = os.path.join(PATH, 'final_project_data.csv')
df = pd.read_csv(path, na_values=['#DIV/0!'])

# Set a random seed to ensure reproducibility across runs
RNG_SEED = 37
np.random.seed(seed=RNG_SEED)

# Drop na's and concatenate two composition components to make a column which 
# can be used to sort the data such that the same composition does not come 
# up in train and test sets
df = df.dropna()
df['comp'] = df['GN'].map(str) + df['PEG'].map(str)

#%%

# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()
  
# Scaling
df[["GN", "PEG", "T"]] = scaler.fit_transform(
    df[["GN", "PEG", "T"]])
df[["GN", "PEG", "T"]] = scaler.fit_transform(
    df[["GN", "PEG", "T"]])

#%%

""" Separate the dataset into training and test sets """

# Store a list of all unique formulae
unique_formulae =  df['comp'].unique()
# Store a list of all unique formulae
all_formulae = unique_formulae.copy()

# Define the proportional size of the dataset split
test_size = 0.20
train_size = 0.80

# Calculate the number of samples in each dataset split
num_test_samples = int(round(test_size * len(unique_formulae)))
num_train_samples = int(round((train_size) * len(unique_formulae)))

# Randomly choose the formulae for the test dataset, and remove those from the unique formulae list
test_formulae = np.random.choice(all_formulae, size=num_test_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in test_formulae]

# The remaining formulae will be used for the training dataset
train_formulae = all_formulae.copy()

# Split the original dataset into the train/validation/test datasets using the formulae lists above
df_train = df[df['comp'].isin(train_formulae)]
df_test = df[df['comp'].isin(test_formulae)]

# make sure data is mutually exclusive (all data with same formula is only in one dataset)
train_formulae = set(df_train['comp'].unique())
test_formulae = set(df_test['comp'].unique())

common_formulae1 = train_formulae.intersection(test_formulae)
common_formulae3 = test_formulae.intersection(test_formulae)

#%%

# saving these splits into csv files
train_path = os.path.join(PATH, 'project_train.csv')
test_path = os.path.join(PATH, 'project_test.csv')

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

#%%

# reading in train/test sets
os.chdir(r'C:\Users\patri\Documents\UofU\MSE7050')
PATH = os.getcwd()
path = os.path.join(PATH, 'project_train.csv')
df_train = pd.read_csv(path, na_values=['#DIV/0!'])
path = os.path.join(PATH, 'project_test.csv')
df_test = pd.read_csv(path, na_values=['#DIV/0!'])

X_train = df_train[['GN','PEG','T']]
X_test = df_test[['GN','PEG','T']]
y_train = df_train[['Conductivity']]
y_test = df_test[['Conductivity']]
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

#%%

#------------------------ gplearn Symbolic Regression -------------------------
"""Best fit = add(sub(-0.547, 0.953), add(add(A, T), sub(div(0.287, -0.136), B)))
log function here is absolute value"""
est_gp = SymbolicRegressor(population_size=10000,
                           generations=300, stopping_criteria=0.01,
                           p_crossover=0.9, p_subtree_mutation=0.01,
                           function_set=('add', 'sub', 'mul', 'div','sqrt','log','cos','sin','tan'),
                           p_hoist_mutation=0.02, p_point_mutation=0.01,
                           max_samples=1.0, verbose=1,tournament_size=30,
                           parsimony_coefficient=0.01,
                           feature_names = ['A','B','T'], random_state=0)
est_gp.fit(X_train, y_train)
grid_pred = est_gp.predict(X_test)
print(est_gp._program)

#%%

# getting result array from gplearn using the best equation, i realized after
# the symbolic regressor has a .predict, but I left this in anyway since it
# done already. I used .predict for the Ax hyperparameter tuning
T = X_test['T']
T = np.array(T.values.tolist())
B = X_test['PEG']
B = np.array(B.values.tolist())
A = X_test['GN']
A = np.array(A.values.tolist())
stuff = np.concatenate(([A],[B],[T]),axis=0)
things = np.linspace(0,len(stuff[0])-1,len(stuff[0]))
things = things.astype(int)
grid_pred = []

for thing in things:
    T = stuff[2,thing]
    B = stuff[1,thing]
    A = stuff[0,thing]
    result = np.add(np.subtract(-0.547, 0.953), np.add(np.add(A, T), np.subtract(np.divide(0.287, -0.136), B)))
    grid_pred = np.append(grid_pred,result)

#%%

#------------------------LINEAR REGRESSION GRIDSEARCH--------------------------
""" best values for each parameter came out as: fit_intercept = True
n_jobs =  """

gridsearch = GridSearchCV(estimator=linear(), cv=5,
                          param_grid={
                              'fit_intercept':['True','False'],
                              'n_jobs':[1,5,10,20,50,100]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Linear Regression MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Linear Regression RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

#%%

#------------------------LASSO REGRESSION GRIDSEARCH---------------------------
""" the best value came out as: 'alpha'=0.001 """

gridsearch = GridSearchCV(estimator=lasso(random_state=4), cv=5,
                          param_grid={
                              'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000],
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Lasso MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Lasso RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

#%%

#--------------------------RIDGE REGRESSION GRIDSEARCH-------------------------
""" the best value came out as: 'alpha'= 1 """

gridsearch = GridSearchCV(estimator=ridge(random_state=4), cv=5,
                          param_grid={
                              'alpha':[0.0001,0.001,0.01,0.1,1,10,100,1000]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Ridge MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Ridge RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

#%%

#--------------------------RANDOM FOREST GRIDSEARCH----------------------------
""" the best values for each parameter came out as: 'max_depth'=10,
'min_samples_leaf'=1, 'min_samples_split'=3, and 'n_estimators'=5 """

gridsearch = GridSearchCV(estimator=rfr(random_state=4), cv=5,
                          param_grid={
                              'n_estimators':[5,10,20,30,50],
                              'max_depth':[2,5,10,15,20],
                              'min_samples_split':[2,3,4,5],
                              'min_samples_leaf':[1,2,3,4,5]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('Random Forest MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('Random Forest RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)
print(abs(grid_result.best_score_))

#%%

#--------------------------SVR GRIDSEARCH--------------------------------------
""" the best values came out as: 'C'=5, 'epsilon'=0.1, and 'gamma'=auto """

gridsearch = GridSearchCV(estimator=SVR(kernel='rbf'), cv=5,
                          param_grid={
                              'gamma':['scale','auto',0.1,1],
                              'C':[5,10,15,20,25,30],
                              'epsilon':[0.0001,0.001,0.01,0.1,1,10,100,1000]
                              },
                          scoring='neg_mean_absolute_error')

grid_result = gridsearch.fit(X_train, y_train)
grid_pred = gridsearch.predict(X_test)
grid_rmse = mean_squared_error(y_test, grid_pred)
print('SVR MAE: ' + str(sum(abs(grid_pred - y_test))/(len(y_test))))
print('SVR RMSE: ' + str(np.sqrt(grid_rmse)))
print(grid_result.best_params_)

#%%

""" I used this block to create parity plots, since the predicted values were 
named the same variable each time I just ran this block to get the same plot
for different regressors """

import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error
r2 = r2_score(y_test, grid_pred)
MAE = mean_absolute_error(y_test, grid_pred)
rms = mean_squared_error(y_test, grid_pred, squared=False)
fig, ax = plt.subplots()

ax.scatter(y_test, grid_pred, c = 'k',edgecolors=(0, 0, 0), marker = 's', label = 'Predicted values')
ax.plot([-1, -5], [-1, -5], "b", lw=1, label = 'Ideal')

#calculate equation for trendline
z = np.polyfit(y_test, grid_pred, 1)
p = np.poly1d(z)
#add trendline to plot
var = np.linspace(-5,-1,40)
ax.plot(var, p(var),  linestyle='--', dashes=(5, 5), color = 'k', lw=1, label = 'Linear Fit')

handles, labels = plt.gca().get_legend_handles_labels()
patch = mpatches.Patch(color='white', label='y = {:.4f}x{:.4f}'.format(z[0],z[1]))   
handles.append(patch)
patch2 = mpatches.Patch(color='white', label= 'R$^2$ = {:.4f}'.format(r2))
handles.append(patch2) 
patch3 = mpatches.Patch(color='white', label='MAE = {:.4f}'.format(MAE)) 
handles.append(patch3) 
patch4 = mpatches.Patch(color='white', label='RMSE = {:.4f}'.format(rms)) 
handles.append(patch4) 
ax.legend(handles=handles)
ax.set_aspect('equal', adjustable='box')
fig.set_size_inches(7, 7)
ax.tick_params(direction='in', length=5, width=1, colors='k')
ax.set_xlim([-3.5, -2.1])
ax.set_ylim([-3.5, -2.1])
ax.grid(visible=True, which='major', axis='both', linestyle = '--', color = 'lightgrey')
ax.set_title('Symbolic Regressor')
ax.set_xlabel("Measured log [$\sigma$ (S$\cdot$cm$^{-1}$)]")
ax.set_ylabel("Predicted log [$\sigma$ (S$\cdot$cm$^{-1}$)]")

#%%

""" This block creates the metrics plot for all the different regressors.
I hard coded in the values from the parity plots since it was easier for me """

# make plot of all the scores for different models

labels = ['Linear', 'Lasso', 'Ridge', 'RF', 'SVR','SR']
R2 = [0.6529, 0.6278, 0.6311, 0.7568, 0.8529,0.8738]
MAE = [0.1902, 0.2012, 0.2000, 0.1500, 0.1325,0.1127]
RMSE = [0.2248,0.2328,0.2317,0.1882,0.1463,0.1355]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
ax2 = ax.twinx() # Create another axis that shares the same x-axis as ax.
rects1 = ax.bar(x - 0.3, R2, width, label='R$^2$',facecolor = 'darkblue',edgecolor = 'lightgrey', hatch = '///')
rects2 = ax2.bar(x, RMSE, width, label='RMSE',facecolor = 'royalblue',edgecolor = 'lightgrey',hatch = '---')
rects3 = ax2.bar(x + 0.3, MAE, width, label='MAE',facecolor = 'lightskyblue',edgecolor = 'lightgrey',hatch = 'xxx')

rects4 = ax.bar(x - 0.3, R2, width,facecolor='none', edgecolor = 'k')
rects5 = ax.bar(x, RMSE, width,facecolor='none', edgecolor = 'k')
rects6 = ax.bar(x+0.3, MAE, width,facecolor='none', edgecolor = 'k')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim([0, 1])
ax2.set_ylim([0,0.4])
ax.set_ylabel('Scores')
ax.set_xticks(x, labels)
ax.tick_params(direction='in', length=5, width=1, colors='k')
ax2.tick_params(direction='in', length=5, width=1, colors='k')
fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)
fig.tight_layout()
ax.set_ylabel("R$^2$ Score")
ax2.set_ylabel("RMSE/MAE Score")
plt.show()

#%%

"""" This block makes the log cond vs 1000/T plot for the predicted values
from the symbolic regressor """

x2 = [3.355705,3.30033,3.194888,3.095975,3.003003,2.915452,2.832861]
ytrue2 = y_test[0:7]
ypred2 = grid_pred[0:7]

fig, ax = plt.subplots()
ax.plot(x2, ytrue2, c = 'k', marker = 's', label = 'Measured')
ax.plot(x2, ypred2, c = 'b',linestyle = '--', marker = '^', label = 'Predicted')

fig.set_size_inches(7, 7)
ax.tick_params(direction='in', length=5, width=1, colors='k')
ax.set_xlim([2.75, 3.45])
ax.set_ylim([-4, -2])
ax.set_box_aspect(1)
ax.set_xlabel("1000 / T")
ax.set_ylabel("log [$\sigma$ (S$\cdot$cm$^{-1}$)]")
fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

#%%

"""" This block makes the surface plot for the symbolic regressor function"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return (np.add(np.subtract(-0.547, 0.953), np.add(np.add(x, 0), np.subtract(np.divide(0.287, -0.136), y)))) # for RT T = 0

# creating data set for surface plot
x = np.linspace(0.1,1,50)
y = np.linspace(0.1,1,50)
x, y = np.meshgrid(x, y)
z = f(x, y)

# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='coolwarm', edgecolor='none')
ax.set_title('T = 25$^{\circ}$C')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('log [$\sigma$ (S$\cdot$cm$^{-1}$)]')
ax.invert_yaxis()

# show plot
plt.show()

#%%

"""" Up until line 504 the code is copied and pasted from the top section
but I created train/val/test splits instead of just train/test for Ax
optimization of the symbolic regressor """

import math
import os
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LinearRegression as linear
from sklearn.linear_model import Lasso as lasso
from sklearn.linear_model import Ridge as ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import graphviz
from sklearn.preprocessing import MinMaxScaler

#get csv and create dataframe, determine initial size
os.chdir(r'C:\Users\patri\Documents\UofU\MSE7050')
PATH = os.getcwd()
path = os.path.join(PATH, 'final_project_data.csv')
df = pd.read_csv(path, na_values=['#DIV/0!'])

# Set a random seed to ensure reproducibility across runs
RNG_SEED = 37
np.random.seed(seed=RNG_SEED)

# Drop na's and concatenate two composition components to make a column which 
# can be used to sort the data such that the same composition does not come 
# up in train and test sets
df = df.dropna()
df['comp'] = df['GN'].map(str) + df['PEG'].map(str)

# Creating an instance of the sklearn.preprocessing.MinMaxScaler()
scaler = MinMaxScaler()
  
# Scaling
df[["GN", "PEG", "T"]] = scaler.fit_transform(
    df[["GN", "PEG", "T"]])
df[["GN", "PEG", "T"]] = scaler.fit_transform(
    df[["GN", "PEG", "T"]])

# Separate the dataset into training and test sets
unique_formulae =  df['comp'].unique()
# Store a list of all unique formulae
all_formulae = unique_formulae.copy()

# Define the proportional size of the dataset split
test_size = 0.10
val_size = 0.10
train_size = 0.80

# Calculate the number of samples in each dataset split
num_test_samples = int(round(test_size * len(unique_formulae)))
num_val_samples = int(round(val_size * len(unique_formulae)))
num_train_samples = int(round((train_size) * len(unique_formulae)))

# Randomly choose the formulate for the validation dataset, and remove those from the unique formulae list
val_formulae = np.random.choice(all_formulae, size=num_val_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in val_formulae]

# Randomly choose the formulae for the test dataset, and remove those from the unique formulae list
test_formulae = np.random.choice(all_formulae, size=num_test_samples, replace=False)
all_formulae = [f for f in all_formulae if f not in test_formulae]

# The remaining formulae will be used for the training dataset
train_formulae = all_formulae.copy()

print('Number of training formulae:', len(train_formulae))
print('Number of validation formulae:', len(val_formulae))
print('Number of testing formulae:', len(test_formulae))

# Split the original dataset into the train/validation/test datasets using the formulae lists above
df_train = df[df['comp'].isin(train_formulae)]
df_val = df[df['comp'].isin(val_formulae)]
df_test = df[df['comp'].isin(test_formulae)]

print(f'train dataset shape: {df_train.shape}')
print(f'validation dataset shape: {df_val.shape}')
print(f'test dataset shape: {df_test.shape}\n')

# make sure data is mutually exclusive (all data with same formula is only in one dataset)
train_formulae = set(df_train['comp'].unique())
val_formulae = set(df_val['comp'].unique())
test_formulae = set(df_test['comp'].unique())

common_formulae1 = train_formulae.intersection(train_formulae)
common_formulae2 = train_formulae.intersection(val_formulae)
common_formulae3 = test_formulae.intersection(test_formulae)

# saving these splits into csv files
train_path = os.path.join(PATH, 'project_train.csv')
val_path = os.path.join(PATH, 'project_val.csv')
test_path = os.path.join(PATH, 'project_test.csv')

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
df_test.to_csv(test_path, index=False)

# reading in train/test sets
os.chdir(r'C:\Users\patri\Documents\UofU\MSE7050')
PATH = os.getcwd()
path = os.path.join(PATH, 'project_train.csv')
df_train = pd.read_csv(path, na_values=['#DIV/0!'])
path = os.path.join(PATH, 'project_test.csv')
df_test = pd.read_csv(path, na_values=['#DIV/0!'])

# splitting the data to x,y train val test sets
X_train = df_train[['GN','PEG','T']]
X_val = df_val[['GN','PEG','T']]
X_test = df_test[['GN','PEG','T']]
y_train = df_train[['Conductivity']]
y_val = df_val[['Conductivity']]
y_test = df_test[['Conductivity']]
y_train = y_train.values.ravel()
y_val = y_val.values.ravel()
y_test = y_test.values.ravel()

#%%

""" Best parameters from training on train dataset:  
{'population_size': 3771, 'generations': 22, 'parsimony_coefficient': 0.006700419636826137,
'tournament_size': 28, 'p_crossover': 0.8034299701247525, 'p_subtree_mutation': 0.01031475532409723,
'p_hoist_mutation': 0.014080560875353226, 'p_point_mutation': 0.011497281665997363}"""

from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor as rfr

# define an evaluation function
def sr_evaluation_function(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(8)])
    sr = SymbolicRegressor(**parameterization) 
    sr.fit(X_train, y_train) # train only on the train dataset
    pred = sr.predict(X_val) # predict only on val dataset
    return {"MAE": mean_absolute_error(y_val,pred)} #used MAE as metric

# create hyperparameter search space, used the same values as for grid search
best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "population_size",
            "type": "range",
            "bounds": [500,5000]
        },
        {
            "name": "generations",
            "type": "range",
            "bounds": [10,100],
        },
        {
            "name": "parsimony_coefficient",
            "type": "range",
            "bounds": [0.0001,0.01],
        },
        {
            "name": "tournament_size",
            "type": "range",
            "bounds": [10,40]
        },
        {
            "name": "p_crossover",
            "type": "range",
            "bounds": [0.7,0.9]
        },
        {
            "name": "p_subtree_mutation",
            "type": "range",
            "bounds": [0.005,0.015]
        },
        {
            "name": "p_hoist_mutation",
            "type": "range",
            "bounds": [0.005,0.015]
        },
        {
            "name": "p_point_mutation",
            "type": "range",
            "bounds": [0.005,0.015]
        }
    ],
    experiment_name="sr_ax",
    objective_name="MAE",
    evaluation_function=sr_evaluation_function,
    minimize=True,
    total_trials=100,
)
print('Best parameters from training on train dataset: ', best_parameters)
means, covariances = values

#%%

#------------------------ gplearn Symbolic Regression -------------------------
""" Best parameters from training on train dataset:  
{'population_size': 3771, 'generations': 22, 'parsimony_coefficient': 0.006700419636826137,
'tournament_size': 28, 'p_crossover': 0.8034299701247525, 'p_subtree_mutation': 0.01031475532409723,
'p_hoist_mutation': 0.014080560875353226, 'p_point_mutation': 0.011497281665997363}"""

# concatenate train and val sets to train on
X_train2 = np.concatenate((X_train,X_val))
y_train2 = np.concatenate((y_train,y_val))

est_gp = SymbolicRegressor(population_size=3771,
                           generations=22,
                           parsimony_coefficient=0.006700419636826137,
                           tournament_size=28,
                           p_crossover=0.8034299701247525,
                           p_subtree_mutation=0.01031475532409723,
                           p_hoist_mutation=0.014080560875353226,
                           p_point_mutation=0.011497281665997363
                          )
est_gp.fit(X_train2, y_train2)
grid_pred = est_gp.predict(X_test)
print(est_gp._program) # add(sub(X0, X1), add(add(add(X2, -0.177), -0.306), div(-0.306, 0.097)))
