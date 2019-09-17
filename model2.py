import csv
import cv2
import numpy as np
lines = []
with open('driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for i,line in enumerate(lines):
    if i > 0:
        source_path = line[0]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = 'data/IMG/' + filename
        image = cv2.imread(local_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(image)
        measurement = float(line[3]) * 1.5 #放大测量
        measurements.append(measurement)
#翻转图片
augmented_images = []
augmented_measurements = []
 
for image,measurement in zip(images,measurements):
    #zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image,1)#1 水平翻转 0 垂直翻转 -1 水平垂直翻转
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.utils.visuallize_util import plot
from keras.utils import plot_model

model = Sequential() #模型接口
model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape = (160,320,3)))#图像归一化
#https://keras-cn.readthedocs.io/en/latest/layers/core_layer/#lambda
model.add(Cropping2D(cropping=((70,25),(0,0))))#裁剪图像
#https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#cropping2d
model.add(Convolution2D(24,5,5,subsample = (2,2),activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# subsample 代表向左和向下的过滤窗口移动步幅
model.add(Convolution2D(36,5,5,subsample = (2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample = (2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
 
model.compile(optimizer = 'adam',loss = 'mse')
model.fit(X_train,y_train,validation_split = 0.2,shuffle = True , nb_epoch = 10)
 
#plot_model(model,to_file = 'model_drop.png',show_shapes = True)
model.summary()
model.save('model-L2-test.h5')
