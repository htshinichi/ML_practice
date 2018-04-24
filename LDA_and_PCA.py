# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:58:29 2018

@author: htshinichi
"""

from mpl_toolkits.mplot3d import Axes3D
##用于可视化图表
import matplotlib.pyplot as plt
##用于加载数据或生成数据等
from sklearn import datasets
##导入PCA库
from sklearn.decomposition import PCA
##导入LDA库
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
iris_X = iris.data   ##获得数据集中的输入
iris_y = iris.target ##获得数据集中的输出，即标签(也就是类别)
print(iris_X.shape)
print(iris.feature_names)
print(iris.target_names)

##加载PCA模型并训练、降维
model_pca = PCA(n_components=4)
X_pca = model_pca.fit(iris_X).transform(iris_X)
print("各主成分方向：\n",model_pca.components_)
print("各主成分的方差值：",model_pca.explained_variance_)
print("各主成分的方差值与总方差之比：",model_pca.explained_variance_ratio_)
print("奇异值分解后得到的特征值：",model_pca.singular_values_)
print("主成分数：",model_pca.n_components_)

##加载PCA模型并训练、降维
model_pca = PCA(n_components=3)
X_pca = model_pca.fit(iris_X).transform(iris_X)
print("降维后各主成分方向：\n",model_pca.components_)
print("降维后各主成分的方差值：",model_pca.explained_variance_)
print("降维后各主成分的方差值与总方差之比：",model_pca.explained_variance_ratio_)
print("奇异值分解后得到的特征值：",model_pca.singular_values_)
print("降维后主成分数：",model_pca.n_components_)

##通过改变elev和azim的值，可以看到不同投影下的散点图情况
fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig,rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], marker='o',c=iris_y)
plt.show()

##加载PCA模型并训练、降维
model_pca = PCA(n_components=2)
X_pca = model_pca.fit(iris_X).transform(iris_X)
print("降维后各主成分方向：\n",model_pca.components_)
print("降维后各主成分的方差值：",model_pca.explained_variance_)
print("降维后各主成分的方差值与总方差之比：",model_pca.explained_variance_ratio_)
print("奇异值分解后得到的特征值：",model_pca.singular_values_)
print("降维后主成分数：",model_pca.n_components_)

fig = plt.figure(figsize=(10,8))
plt.scatter(X_pca[:, 0], X_pca[:, 1],marker='o',c=iris_y)
plt.show()

#加载LDA模型并训练，降维
model_lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = model_lda.fit(iris_X, iris_y).transform(iris_X)
print("降维后各主成分的方差值与总方差之比：",model_lda.explained_variance_ratio_)
print(model_lda.classes_)
print("降维前样本数量和维度：",iris_X.shape)
print("降维后样本数量和维度：",X_lda.shape)

fig = plt.figure(figsize=(10,8))
plt.scatter(X_lda[:, 0], X_lda[:, 1],marker='o',c=iris_y)
plt.show()
