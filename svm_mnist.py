import numpy as np
import struct
import matplotlib.pyplot as plt
import os
##加载svm模型
from sklearn import svm
###用于做数据预处理
from sklearn import preprocessing
import time

path='E:\\Desktop\\code_practice\\DATA'
def load_mnist_train(path, kind='train'):    
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels
def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    return images, labels   
train_images,train_labels=load_mnist_train(path)
test_images,test_labels=load_mnist_test(path)

X=preprocessing.StandardScaler().fit_transform(train_images)  
X_train=X[0:60000]
y_train=train_labels[0:60000]

print(time.strftime('%Y-%m-%d %H:%M:%S'))
model_svc = svm.LinearSVC()
#model_svc = svm.SVC()
model_svc.fit(X_train,y_train)
print(time.strftime('%Y-%m-%d %H:%M:%S'))

##显示前30个样本的真实标签和预测值，用图显示
x=preprocessing.StandardScaler().fit_transform(test_images)
x_test=x[0:10000]
y_pred=test_labels[0:10000]
print(model_svc.score(x_test,y_pred))
y=model_svc.predict(x)

fig1=plt.figure(figsize=(8,8))
fig1.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(100):
    ax=fig1.add_subplot(10,10,i+1,xticks=[],yticks=[])
    ax.imshow(np.reshape(test_images[i], [28,28]),cmap=plt.cm.binary,interpolation='nearest')
    ax.text(0,2,"pred:"+str(y[i]),color='red')
    #ax.text(0,32,"real:"+str(test_labels[i]),color='blue')
plt.show()
