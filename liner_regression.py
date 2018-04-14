# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 12:31:08 2018

@author: htshinichi
"""
##用于可视化图表
import matplotlib.pyplot as plt
##用于做科学计算
import numpy as np
##用于做数据分析
import pandas as pd
##用于加载数据或生成数据等
from sklearn import datasets
##加载线性模型
from sklearn import linear_model
###用于交叉验证以及训练集和测试集的划分
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
###这个模块中含有评分函数，性能度量，距离计算等
from sklearn import metrics


boston = datasets.load_boston()
print(boston.data.shape)
print(boston["feature_names"])

boston_X = boston.data   ##获得数据集中的输入
boston_y = boston.target ##获得数据集中的输出，即标签(也就是类别)
boston_data = pd.DataFrame(boston_X)
boston_data.columns = boston.feature_names
boston_data["PRICE"]=boston_y
boston_data.head(10)

data_X=boston_data[['ZN','RM','PTRATIO','LSTAT']]
data_X.head(10)

data_y=boston_data[['PRICE']]
data_y.head(10)

### test_size:测试数据大小
X_train,X_test,y_train,y_test = train_test_split(data_X, data_y, test_size = 0.1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

##加载线性回归模型
model=linear_model.LinearRegression()
##将训练数据传入开始训练
model.fit(X_train,y_train)
print(model.coef_)     #系数，有些模型没有系数（如k近邻）
print(model.intercept_) #与y轴交点，即截距

model_ridge=linear_model.Ridge(alpha =100000)
model_ridge.fit(X_train,y_train)
model_lasso=linear_model.Lasso(alpha=10)
model_lasso.fit(X_train,y_train)
print(model_ridge.coef_)     #系数，有些模型没有系数（如k近邻）
print(model_ridge.intercept_) #与y轴交点，即截距
print(model_lasso.coef_)     #系数，有些模型没有系数（如k近邻）
print(model_lasso.intercept_) #与y轴交点，即截距

y_pred = model.predict(X_test)
y_ridge_pred = model_ridge.predict(X_test)
y_lasso_pred = model_lasso.predict(X_test)
print("使用LinearRegression模型的均方误差为:",metrics.mean_squared_error(y_test, y_pred))
print("使用LinearRegression模型的均方根误差为:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("使用Ridge Regression模型的均方误差为:",metrics.mean_squared_error(y_test, y_ridge_pred))
print("使用Ridge Regression模型的均方根误差为:",np.sqrt(metrics.mean_squared_error(y_test, y_ridge_pred)))
print("使用Lasso Regression模型的均方误差为:",metrics.mean_squared_error(y_test, y_lasso_pred))
print("使用Lasso Regression模型的均方根误差为:",np.sqrt(metrics.mean_squared_error(y_test, y_lasso_pred)))

predicted = cross_val_predict(model, data_X, data_y, cv=10)
print("使用交叉验证的均方误差为:",metrics.mean_squared_error(data_y, predicted))
print(model.score(X_test,y_test))
print(model_ridge.score(X_test,y_test))
print(model_lasso.score(X_test,y_test))

plt.figure('model')
plt.plot(data_y,predicted,'.')
plt.plot([data_y.min(),data_y.max()],[data_y.min(),data_y.max()],'k--',lw=2)
plt.scatter(data_y,predicted)
plt.show()

alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
scores1 = []
scores2 = []
for i, alpha in enumerate(alphas):
    model_ridge = linear_model.Ridge(alpha=alpha)
    model_ridge.fit(X_train, y_train)
    scores1.append(model_ridge.score(X_test, y_test))
    model_lasso = linear_model.Lasso(alpha=alpha)
    model_lasso.fit(X_train, y_train)
    scores2.append(model_lasso.score(X_test, y_test))
figure = plt.figure(figsize=(8,6))
ax = figure.add_subplot(1, 1, 1)
ax.plot(alphas, scores1,color='red',lw=1,label='Ridge')
ax.plot(alphas, scores2,lw=1,label='Lasso')
plt.legend(loc='upper right',frameon=False)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"score")
ax.set_xscale("log")
ax.set_title("Ridge&Lasso")
plt.show()
