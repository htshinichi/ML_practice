# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:21:43 2018

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
#from sklearn.cross_validation import cross_val_score
###这个模块中含有评分函数，性能度量，距离计算等
from sklearn import metrics
###用于做数据预处理
from sklearn import preprocessing

digits = datasets.load_digits()
print(digits.keys())

print(digits.data.shape)
print(digits.target_names)
for i in range(0,2):
    print(digits.target[i])
    print(digits.images[i])
    print(digits.data[i])

plt.gray()
for i in range(0,2):
    plt.matshow(digits.images[i])
    plt.show()
    print(digits.target[i])
fig=plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

#绘制数字：每张图像8*8像素点
for i in range(30):
    ax=fig.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    #用目标值标记图像
    ax.text(0,7,str(digits.target[i]))
plt.show()

digits_X = digits.data   ##获得数据集中的输入
digits_y = digits.target ##获得数据集中的输出，即标签(也就是类别)
### test_size:测试数据大小
X_train,X_test,y_train,y_test = train_test_split(digits_X, digits_y, test_size = 0.1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
plt.gray()
plt.matshow(np.abs(X_train[0].reshape(8, 8)))
plt.show()
X_train=preprocessing.StandardScaler().fit_transform(X_train)

plt.gray()
plt.matshow(np.abs(X_train[0].reshape(8, 8)))
plt.show()

print(X_train[0])
print("第一个特征的均值为",X_train[0].mean())
print("第一个特征的方差为",X_train[0].var())    
print(y_train[0:20])
y_train_bin = (y_train > 4).astype(np.int)
print(y_train_bin[0:20])


##加载逻辑回归模型
model_LR_l1=linear_model.LogisticRegression(C=0.5, penalty='l1', tol=0.01)
##将训练数据传入开始训练
model_LR_l1.fit(X_train,y_train_bin)
X_test_scale=preprocessing.StandardScaler().fit_transform(X_test)
y_test_bin = (y_test>4).astype(np.int)
print(model_LR_l1.score(X_test_scale,y_test_bin))
y_pred = model_LR_l1.predict(X_test_scale)
##显示前30个样本的真实标签和预测值，用图显示
fig1=plt.figure(figsize=(8,8))
fig1.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(30):
    ax=fig1.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,1,str(y_test[i]))
    ax.text(0,7,str(y_pred[i]))
plt.show()
##找出分类错误的样本，用图显示
fig2=plt.figure(figsize=(8,8))
fig2.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
num=0
for i in range(180):
    if(y_test_bin[i]!=y_pred[i]):
        num=num+1
        ax=fig2.add_subplot(12,5,num,xticks=[],yticks=[])
        ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')
        ax.text(0,1,str(y_test[i]))
        ax.text(0,7,str(y_pred[i]))
plt.show()
print(num)


##加载逻辑回归模型，选择随机平均梯度下降，多分类方法用one vs rest
model_LR_ovr=linear_model.LogisticRegression(solver='sag',max_iter=3000,random_state=42,multi_class='ovr')
##将训练数据传入开始训练
model_LR_ovr.fit(X_train,y_train)
X_test_scale=preprocessing.StandardScaler().fit_transform(X_test)
print(model_LR_ovr.score(X_test_scale,y_test))
y_pred = model_LR_ovr.predict(X_test_scale)
##显示前30个样本的真实标签和预测值，用图显示
fig3=plt.figure(figsize=(8,8))
fig3.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(30):
    ax=fig3.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,1,str(y_test[i]))
    ax.text(0,7,str(y_pred[i]))
plt.show()
##找出分类错误的样本，用图显示
fig4=plt.figure(figsize=(8,8))
fig4.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
num=0
for i in range(180):
    if(y_test[i]!=y_pred[i]):
        num=num+1
        ax=fig4.add_subplot(6,5,num,xticks=[],yticks=[])
        ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')

        ax.text(0,1,str(y_test[i]))
        ax.text(0,7,str(y_pred[i]))
plt.show()
print(num)


##加载逻辑回归模型，选择随机平均梯度下降，多分类方法用many vs many
model_LR_mult=linear_model.LogisticRegression(solver='sag',max_iter=3000,random_state=42,multi_class='multinomial')
##将训练数据传入开始训练
model_LR_mult.fit(X_train,y_train)
X_test_scale=preprocessing.StandardScaler().fit_transform(X_test)
print(model_LR_mult.score(X_test_scale,y_test))
y_pred = model_LR_mult.predict(X_test_scale)
##显示前30个样本的真实标签和预测值，用图显示
fig5=plt.figure(figsize=(8,8))
fig5.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(30):
    ax=fig5.add_subplot(6,5,i+1,xticks=[],yticks=[])
    ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,1,str(y_test[i]))
    ax.text(0,7,str(y_pred[i]))
plt.show()
##找出分类错误的样本，用图显示
fig6=plt.figure(figsize=(8,8))
fig6.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
num=0
for i in range(180):
    if(y_test[i]!=y_pred[i]):
        num=num+1
        ax=fig6.add_subplot(6,5,num,xticks=[],yticks=[])
        ax.imshow(np.abs(X_test[i].reshape(8, 8)),cmap=plt.cm.binary,interpolation='nearest')
        #用目标值标记图像
        ax.text(0,1,str(y_test[i]))
        ax.text(0,7,str(y_pred[i]))
plt.show()
print(num)
