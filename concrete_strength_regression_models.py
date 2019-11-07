# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:27:18 2019

@author: Sravanii
"""

import pandas as pd
import numpy as np
df=pd.read_excel('Case_3_LR_manufacturing.xls')
df.columns=['Cement','Blast_Furnace_Slag','Fly_Ash','Water','Superplasticizer','Coarse_Aggregate','Fine_Aggregate','Age','Concrete_Strength']

df.describe()
df.info()

df.isnull().sum()
'''
df.boxplot(column=['Concrete_Strength'])
df.boxplot()'''

################# Outliers ##################################################
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1

index_outliers=df.loc[((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)].index.tolist()
list_outliers=df.loc[index_outliers]

remove_outliers=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

df_final=remove_outliers

############################################################################


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(df_final,test_size=0.2,random_state=10 )

X_train=train_set.iloc[:,0:-1].values
y_train=train_set.iloc[:,-1].values

X_test=test_set.iloc[:,0:-1].values
y_test=test_set.iloc[:,-1].values

##################### Regression Models #####################################
##################### Call and Fitting the models ############################
#################### Linear Regression #######################################
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)

##############################################################################

##################### Ridge ##################################################
from sklearn.linear_model import Ridge
ridge_reg=Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X_train,y_train)

#############################################################################

################## Lasso ####################################################
from sklearn.linear_model import Lasso
lasso_reg=Lasso(alpha=0.1)
lasso_reg.fit(X_train,y_train)

############################################################################

################ Elastic Net ###############################################
from sklearn.linear_model import ElasticNet
elastic_net_reg=ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_reg.fit(X_train,y_train)

#############################################################################

################# SGD Regressor #############################################
from sklearn.linear_model import SGDRegressor
SGD_reg=SGDRegressor()
SGD_reg.fit(X_train,y_train)

#############################################################################

################## Decision Tree#############################################
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train,y_train)

#############################################################################

##################Random Forest #############################################
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train,y_train)

############################################################################

################ XG Boost ##################################################
import xgboost as xgb
xg_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, silent=False)
xg_reg.fit(X_train,y_train)

#############################################################################

################## SVM ######################################################
from sklearn.svm import SVR
svm_reg =SVR(kernel='linear',gamma='auto')
svm_reg.fit(X_train, y_train)
   

##################### RMSE for Train Data ###################################
#################### Linear ###############################################
from sklearn.metrics import mean_squared_error
y_pred_train=lin_reg.predict(X_train)
lin_reg_mse=mean_squared_error(y_train,y_pred_train)
lin_reg_rmse=np.sqrt(lin_reg_mse)
lin_reg_rmse

############################################################################

###################### Ridge #################################################
y_pred_ridge_train=ridge_reg.predict(X_train)
ridge_reg_mse=mean_squared_error(y_train,y_pred_ridge_train)
ridge_reg_rmse=np.sqrt(ridge_reg_mse)
ridge_reg_rmse

#########################################################################

############## Lasso ####################################################
y_pred_lasso_train=lasso_reg.predict(X_train)
lasso_reg_mse=mean_squared_error(y_train,y_pred_lasso_train)
lasso_reg_rmse=np.sqrt(lasso_reg_mse)
lasso_reg_rmse

###########################################################################

################ Elastic Net ##############################################
y_pred_elastic_net_train=elastic_net_reg.predict(X_train)
elastic_net_reg_mse=mean_squared_error(y_train,y_pred_elastic_net_train)
elastic_net_reg_rmse=np.sqrt(elastic_net_reg_mse)
elastic_net_reg_rmse

##########################################################################

################## SGD ####################################################
y_pred_SGD_train=SGD_reg.predict(X_train)
SGD_reg_mse=mean_squared_error(y_train,y_pred_SGD_train)
SGD_reg_rmse=np.sqrt(SGD_reg_mse)
SGD_reg_rmse

#######################################################################

############### Decision Tree #########################################
y_pred_tree_train=tree_reg.predict(X_train)
tree_reg_mse=mean_squared_error(y_train,y_pred_tree_train)
tree_reg_rmse=np.sqrt(tree_reg_mse)
tree_reg_rmse

######################################################################

################ Random Forest #####################################
y_pred_forest_train=forest_reg.predict(X_train)
forest_reg_mse=mean_squared_error(y_train,y_pred_forest_train)
forest_reg_rmse=np.sqrt(forest_reg_mse)
forest_reg_rmse

######################################################################

############### XG Boost ##############################################
y_pred_xg_train=xg_reg.predict(X_train)
xg_reg_mse=mean_squared_error(y_train,y_pred_xg_train)
xg_reg_rmse=np.sqrt(xg_reg_mse)
xg_reg_rmse

############# SVM ####################################################
y_pred_svm_train=svm_reg.predict(X_train)
svm_reg_mse=mean_squared_error(y_train,y_pred_svm_train)
svm_reg_rmse=np.sqrt(svm_reg_mse)
svm_reg_rmse

train_rmse=[lin_reg_rmse,ridge_reg_rmse,lasso_reg_rmse,elastic_net_reg_rmse,SGD_reg_rmse,tree_reg_rmse,forest_reg_rmse,xg_reg_rmse,svm_reg_rmse]
aa=pd.DataFrame(train_rmse)


########################################################################

####################### RMSE for Test data ################################
y_pred_test=lin_reg.predict(X_test)
lin_mse_test=mean_squared_error(y_test,y_pred_test)
lin_rmse_test=np.sqrt(lin_mse_test)
lin_rmse_test

y_pred_ridge_test=ridge_reg.predict(X_test)
ridge_mse_test=mean_squared_error(y_test,y_pred_ridge_test)
ridge_rmse_test=np.sqrt(ridge_mse_test)
ridge_rmse_test

y_pred_lasso_test=lasso_reg.predict(X_test)
lasso_mse_test=mean_squared_error(y_test,y_pred_lasso_test)
lasso_rmse_test=np.sqrt(lasso_mse_test)
lasso_rmse_test

y_pred_elastic_net_test=elastic_net_reg.predict(X_test)
elastic_net_mse_test=mean_squared_error(y_test,y_pred_elastic_net_test)
elastic_net_rmse_test=np.sqrt(elastic_net_mse_test)
elastic_net_rmse_test

y_pred_SGD_test=SGD_reg.predict(X_test)
SGD_mse_test=mean_squared_error(y_test,y_pred_SGD_test)
SGD_rmse_test=np.sqrt(SGD_mse_test)
SGD_rmse_test

y_pred_tree_test=tree_reg.predict(X_test)
tree_mse_test=mean_squared_error(y_test,y_pred_tree_test)
tree_rmse_test=np.sqrt(tree_mse_test)
tree_rmse_test

y_pred_forest_test=forest_reg.predict(X_test)
forest_mse_test=mean_squared_error(y_test,y_pred_forest_test)
forest_rmse_test=np.sqrt(forest_mse_test)
forest_rmse_test

y_pred_xg_test=xg_reg.predict(X_test)
xg_mse_test=mean_squared_error(y_test,y_pred_xg_test)
xg_rmse_test=np.sqrt(xg_mse_test)
xg_rmse_test

y_pred_svm_test=svm_reg.predict(X_test)
svm_mse_test=mean_squared_error(y_test,y_pred_svm_test)
svm_rmse_test=np.sqrt(svm_mse_test)
svm_rmse_test

test_rmse=[lin_rmse_test,ridge_rmse_test,lasso_rmse_test,elastic_net_rmse_test,SGD_rmse_test,tree_rmse_test,forest_rmse_test,xg_rmse_test,svm_rmse_test]
aa1=pd.DataFrame(test_rmse)

###########################################################################

################# R^2 Score for train data #############################

R2_lin_train=lin_reg.score(X_train,y_train)
R2_ridge_train=ridge_reg.score(X_train,y_train)
R2_lasso_train=lasso_reg.score(X_train,y_train)
R2_elastic_net_train=elastic_net_reg.score(X_train,y_train)
R2_SGD_train=SGD_reg.score(X_train,y_train)
R2_tree_train=tree_reg.score(X_train,y_train)
R2_forest_train=forest_reg.score(X_train,y_train)
R2_xg_train=xg_reg.score(X_train,y_train)
R2_svm_train=svm_reg.score(X_train,y_train)

train_r2=[R2_lin_train,R2_ridge_train,R2_lasso_train,R2_elastic_net_train,R2_SGD_train,R2_tree_train,R2_forest_train,R2_xg_train,R2_svm_train]
aa2=pd.DataFrame(train_r2)

##############################################################################

################### R^2 Score for test data #################################

R2_lin_test=lin_reg.score(X_test,y_test)
R2_ridge_test=ridge_reg.score(X_test,y_test)
R2_lasso_test=lasso_reg.score(X_test,y_test)
R2_elastic_net_test=elastic_net_reg.score(X_test,y_test)
R2_SGD_test=SGD_reg.score(X_test,y_test)
R2_tree_test=tree_reg.score(X_test,y_test)
R2_forest_test=forest_reg.score(X_test,y_test)
R2_xg_test=xg_reg.score(X_test,y_test)
R2_svm_test=svm_reg.score(X_test,y_test)

test_r2=[R2_lin_test,R2_ridge_test,R2_lasso_test,R2_elastic_net_test,R2_SGD_test,R2_tree_test,R2_forest_test,R2_xg_test,R2_svm_test]
aa3=pd.DataFrame(test_r2)
#######################################################################################

model=['Linear','Ridge','Lasso','Elastic_Net','SGD','Decision Tree','Random Forest','XG Boost','SVM']
aa4=pd.DataFrame(model)
Result=pd.concat([aa4,aa,aa1,aa2,aa3],axis=1)
Result.columns=['MODEL','RMSE_TRAIN','RMSE_TEST','R^2_SCORE_TRAIN','R^2_SCORE_TEST']
Result.to_csv("Concrete_Strength.csv",index=False)

