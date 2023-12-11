#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AeroCNN-I
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import datetime


# In[27]:


n_units=512
l2Regularizer=1e-09
kernel_size = 2
n_grid = 101
strides = 1
input_size = 49


# In[2]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[3]:


mirrored_strategy = tf.distribute.MirroredStrategy(["/GPU:0","/GPU:1", "/GPU:2"], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())


# In[4]:


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    
    return [x.name for x in local_device_protos]
get_available_gpus()


# In[6]:


alpha = np.array([-10, -2, -4, -6, -8, 0, 10, 12, 14, 16, 18, 20, 2, 4, 6, 8])


# In[7]:


alpha = alpha.reshape((16, 1))


# In[8]:


aa = np.zeros((16*113,1))
for i in range(0, 113):
    aa[16*i:16*(i+1), :] = alpha[:,:]


# In[9]:


aa.shape


# In[10]:


aa = aa.reshape((113*16,1,1))


# In[11]:


os.chdir('D:\\')


# In[12]:


data = pd.read_csv('datasetList.csv', header=None)


# In[13]:


x = data.iloc[:, 0].values


# In[14]:


x = x.reshape((1,113))
x = np.vstack((x,x,x,x,x,x,x,x,
               x,x,x,x,x,x,x,x))
x = x.T.reshape((113*16,1))


# In[15]:


os.chdir('D:\\airfoilInputs')


# In[16]:


path = 'D:\\airfoilInputs\\'
file_list = os.listdir('D:\\airfoilInputs')
file_list_py = [file for file in file_list if file.endswith('.dat')]
df = pd.DataFrame()
for i in file_list_py:
    data = pd.read_table(path + i, sep='\s+', header=None)
    print(data.shape)
    df = pd.concat([df, data], axis=0)


# In[17]:


out = df.iloc[:, :].values


# In[18]:


out.shape # 49(point 개수))*113(데이터셋 개수)//2: x, y coordinate


# In[19]:


out = np.vstack((out,out,out,out,out,out,out,out,
                 out,out,out,out,out,out,out,out))


# In[20]:


out = out.reshape((113*16,49,2,1))


# In[21]:


os.chdir('D:\\airfoilOutputs')


# In[22]:


path = 'D:\\airfoilOutputs\\'
file_list = os.listdir('D:\\airfoilOutputs')
file_list_py = [file for file in file_list if file.endswith('.csv')]
df = pd.DataFrame()
for i in file_list_py:
    data = pd.read_csv(path + i, header=None)
    df = pd.concat([df, data], axis=0)


# In[23]:


y_imp = df.iloc[:, :].values


# In[24]:


y = y_imp.reshape((113*16,n_grid,n_grid, 1))


# In[25]:


x_train, x_test, aa_train, aa_test, geo_train, geo_test, y_train, y_test = train_test_split(out, aa, x, y, test_size=0.3, random_state=1)


# In[29]:
os.chdir('D:\\TrainedModels')

with mirrored_strategy.scope():
    input_coord = tf.keras.Input(shape=(input_size,2,1))
    input_alpha = tf.keras.Input(shape=(1,1))
    #reshape1 = tf.keras.layers.Reshape((input_size,))(input_coord)

    x_conv = tf.keras.layers.Conv2D(n_units, (kernel_size, kernel_size), strides=(strides, strides), activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                    name='Conv2DLayer')(input_coord)
    x_ = tf.keras.layers.MaxPooling2D((2, 1))(x_conv)
    reshape1 = tf.keras.layers.Flatten()(x_)
    reshape2 = tf.keras.layers.Reshape((1,))(input_alpha)
    x_concat = tf.keras.layers.Concatenate(axis=1)([reshape1, reshape2])

    x_ = tf.keras.layers.Dense(units=n_units, activation='relu', name='firstHiddenLayerSensor',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(x_concat)
    x_ = tf.keras.layers.Dense(units=n_units, activation='relu', name='secondHiddenLayerSensor',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(x_)
    x_ = tf.keras.layers.Dense(units=n_units, activation='relu', name='thirdHiddenLayerSensor',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(x_)
    x_ = tf.keras.layers.Dense(units=n_units, activation='relu', name='fourthHiddenLayerSensor',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(x_)
    x_ = tf.keras.layers.Dense(units=n_units, activation='relu', name='fifthHiddenLayerSensor',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(x_)
    output_data = tf.keras.layers.Dense(units=n_grid*n_grid, activation=None, name='outputLayerSensor')(x_)
    output_image = tf.keras.layers.Reshape((n_grid, n_grid))(output_data)
    # AeroCNN-I
    sd = tf.keras.Model([input_coord, input_alpha], output_image)

    sd.summary()

    sd.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss = tf.keras.losses.MeanSquaredError(),
               metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    start = datetime.datetime.now()
    history = sd.fit([x_train,aa_train], y_train, epochs=5000, shuffle=True)
    end = datetime.datetime.now()

    time = end - start
    print("Training time:", time)

    hist = history.history
    plt.plot(hist['loss'], lw=2)
    plt.title('Training loss (mean squared error)', size=15)
    plt.xlabel('Epoch', size=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()
    
    plt.plot(hist['rmse'], lw=2)
    plt.title('Root Mean Squared Error', size=15)
    plt.xlabel('Epoch', size=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

    sd.save('AeroCNN_I_512kernels1StrideMaxpoolingValidCNNpoolingSame.h5', overwrite=True, include_optimizer=True, save_format='h5')

    test_results = sd.evaluate([x_train,aa_train])
    decoded_train = sd.predict([x_train,aa_train]).reshape((len(aa_train),101,101,1))
    decoded_test = sd.predict([x_test, aa_test]).reshape((len(aa_test),101,101,1))
    error_train_abs = np.abs(decoded_train - y_train)
    error_test_abs = np.abs(decoded_test - y_test)


# In[ ]:


l2_error_train = np.sqrt(np.sum((decoded_train - y_train)**2) / np.sum(y_train**2))
print(l2_error_train)


# In[ ]:


l2_error_test = np.sqrt(np.sum((decoded_test - y_test)**2) / np.sum(y_test**2))
print(l2_error_test)


# In[ ]:


l2_error_train_list = []
for i in range(0, len(aa_train)):
    l2_error_train_data = np.sqrt(np.sum((decoded_train[i] - y_train[i])**2) / np.sum(y_train[i]**2))
    l2_error_train_list.append(l2_error_train_data)
#print(l2_error_train_list)


# In[ ]:


l2_error_test_list = []
for i in range(0, len(aa_test)):
    l2_error_test_data = np.sqrt(np.sum((decoded_test[i] - y_test[i])**2) / np.sum(y_test[i]**2))
    l2_error_test_list.append(l2_error_test_data)
#print(l2_error_test_list)


# In[ ]:




