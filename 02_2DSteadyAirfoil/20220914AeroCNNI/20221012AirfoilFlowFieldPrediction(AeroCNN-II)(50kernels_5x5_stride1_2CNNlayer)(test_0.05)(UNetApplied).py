#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AeroCNN-II with U-Net
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
from tensorflow.python.client import device_lib

import datetime

# In[2]:


device_lib.list_local_devices()


# In[3]:


mirrored_strategy = tf.distribute.MirroredStrategy(["/GPU:0","/GPU:1","/GPU:2"], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())


# In[4]:


n_kernel=50
l2Regularizer=1e-09
kernel_size1 = 3
kernel_size2 = 2
#kernel_size3 = 5
n_grid = 101
strides = 1
input_size = 100


# In[5]:


alpha = np.linspace(-10,20,16).reshape((16,1))


# In[6]:


aa = np.zeros((16*133,1))
for i in range(0, 133):
    aa[16*i:16*(i+1), :] = alpha[:,:]


# In[7]:


aa = aa.reshape((133, 16, 1, 1))


# In[8]:


os.chdir('D:\\AeroCNN2Inputs')


# In[9]:


origin = "D:\\AeroCNN2Inputs"
origin_data = "D:\\AirfoilClCdCoordinates_out\\AirfoilClCdCoordinates_out"
origin_coord = "D:\\AirfoilClCdCoordinates_out\\AirfoilClCdCoordinates_out"


# In[10]:


folders_orig = os.listdir(origin)
folders = [file for file in folders_orig if file.endswith('.csv')]


# In[11]:


image_df = pd.DataFrame()
for i in range(1, 134):
    for j in range(0, alpha.shape[0]):
        csv_file_name = origin + '\\airfoil' + str(i) + "_alpha"+ str(int(alpha[j])) + ".csv"
        data = pd.read_csv(csv_file_name, header=None)
        image_df = pd.concat([image_df, data], axis=0)


# In[12]:


image_np = image_df.iloc[:, :].values


# In[13]:


image = image_np.reshape((133, 16, 100, 100))


# In[14]:


image = 1-image/100


# In[15]:


path = 'D:\\airfoilFlowField'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]


# In[16]:


file_name = origin_data + "\\AirfoilIndexList.xlsx"
airfoilName_df = pd.read_excel(file_name)
geometry_orig = airfoilName_df.iloc[:, 0].values


# In[17]:


data_name = path + '\\' + str(geometry_orig[0]) + "alpha"+ str(int(alpha[1])) + "_interpolated.csv"
print(data_name)


# In[18]:


Vmag_df = pd.DataFrame()
for i in range(1, 134):
    for j in range(0, alpha.shape[0]):
        data_name = path + '\\' + str(geometry_orig[i-1]) + "alpha"+ str(int(alpha[j])) + "_interpolated.csv"
        data = pd.read_csv(data_name, header=None)
        Vmag_df = pd.concat([Vmag_df, data], axis=0)


# In[19]:


geometry_orig = airfoilName_df.iloc[:, 1].values


# In[20]:


y_imp = Vmag_df.iloc[:, :].values


# In[21]:


y = y_imp.reshape((133, 16, n_grid, n_grid))


# In[22]:


geometry = np.zeros((133*16,1))
geometry = geometry.astype(np.string_)
for i in geometry_orig:
    index_ = np.where(geometry_orig==i)
    for j in range(0,16):
        geometry[16*index_[0]+j,:] = np.asarray(i)


# In[23]:


geometry.shape


# In[24]:


geometry = geometry.reshape((133, 16, 1))


# In[25]:


x_train, x_test, aa_train, aa_test, geo_train, geo_test, y_train, y_test = train_test_split(image, aa, geometry, y, test_size=0.05, random_state=1)


# In[26]:


x_train = x_train.reshape((x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))
x_test = x_test.reshape((x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3], 1))
aa_train = aa_train.reshape((aa_train.shape[0]*aa_train.shape[1], aa_train.shape[2], aa_train.shape[3]))
aa_test = aa_test.reshape((aa_test.shape[0]*aa_test.shape[1], aa_test.shape[2], aa_test.shape[3]))
geo_train = geo_train.reshape((geo_train.shape[0]*geo_train.shape[1], geo_train.shape[2]))
geo_test = geo_test.reshape((geo_test.shape[0]*geo_test.shape[1], geo_test.shape[2]))
y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
y_test = y_test.reshape((y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))


# In[27]:


with mirrored_strategy.scope():
    input_image = tf.keras.Input(shape=(input_size, input_size, 1))

    x_conv_E1 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                      activation='relu', padding='same',
                                      kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                      name='Conv2DLayer1')(input_image)
    x_conv_E2 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                      activation='relu', padding='same',
                                      kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                      name='Conv2DLayer2')(x_conv_E1) # 100*100
    x_pool_E = tf.keras.layers.MaxPooling2D((2,2))(x_conv_E2)

    x_conv_D0 = tf.keras.layers.Conv2DTranspose(n_kernel, (kernel_size2, kernel_size2), strides = (2, 2),
                                         padding='same', activation='relu')(x_pool_E)
    x_concat = tf.concat([x_conv_E2, x_conv_D0], axis=-1)
    x_conv_D1 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                         activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                         name='DeConv2DLayer1')(x_concat)
    x_conv_D2 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                         activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                         name='DeConv2DLayer2')(x_conv_D1)
    x_conv_1 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                         activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                         name='Conv2DLayerf1')(x_conv_D2)
    x_conv_2 = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                         activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                         name='Conv2DLayerf2')(x_conv_1)

    x_conv_Final = tf.keras.layers.Conv2D(1, (kernel_size1, kernel_size1), strides=(strides, strides),
                                          activation='relu', padding='same',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                          name='FinalConv2DLayer')(x_conv_2)
    reshape1 = tf.keras.layers.Flatten()(x_conv_Final)
    x_ = tf.keras.layers.Dense(units=n_grid*n_grid, activation=None, name='outputLayer',
                               kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer))(reshape1)

    output_image = tf.keras.layers.Reshape((n_grid, n_grid, 1))(x_)
    # AeroCNN-II based
    model = tf.keras.Model(input_image, output_image)


    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    start = datetime.datetime.now()
    history = model.fit(x_train, y_train, epochs=5000, shuffle=True,callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30))
    end = datetime.datetime.now()


# In[31]:


time = end - start
print("Training time:", time)


# In[ ]:


os.chdir("D:\\AeroCNNII_Images")


# In[32]:


hist = history.history
plt.plot(hist['loss'], lw=2)
plt.title('Training loss (mean squared error)\nAeroCNN-II with U-net', size=15)
plt.xlabel('Epoch', size=15)
plt.yscale('log')
#plt.ylim([5e-5, 1e-1])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.savefig("20221012AeroCNNIIwithUnet(2CNNlayer_3x3_1block)(Trainingloss).jpg", dpi=300)
plt.show()


# In[33]:


plt.plot(hist['rmse'], lw=2)
plt.title('Root Mean Squared Error', size=15)
plt.xlabel('Epoch', size=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()


# In[34]:


plt.plot(hist['rmse'], lw=2)
plt.title('Root Mean Squared Error', size=15)
plt.xlabel('Epoch', size=15)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.savefig("20221012AeroCNNIIwithUnet(2CNNlayer_3x3_1block)(RMSE).jpg", dpi=300)
plt.show()


# In[35]:


with mirrored_strategy.scope():
    test_results = model.evaluate(x_train)
    decoded_train = model.predict(x_train)
    decoded_test = model.predict(x_test)


# In[36]:


test_results2 = model.evaluate(x_test)


# In[42]:


decoded_train = decoded_train.reshape((int(y_train.shape[0]),int(n_grid),int(n_grid)))
decoded_test = decoded_test.reshape((int(y_test.shape[0]),int(n_grid),int(n_grid)))


# In[45]:


error_train_abs = np.abs(decoded_train - y_train)


# In[46]:


error_test_abs = np.abs(decoded_test - y_test)


# In[47]:


error_train_Cl_abs = np.abs(decoded_train[:,0,:] - y_train[:,0,:])


# In[48]:


error_train_Cd_abs = np.abs(decoded_train[:,1,:] - y_train[:,1,:])


# In[49]:


error_test_Cl_abs = np.abs(decoded_test[:,0,:] - y_test[:,0,:])


# In[50]:


error_test_Cd_abs = np.abs(decoded_test[:,1,:] - y_test[:,1,:])


# In[51]:


decoded_train.shape


# In[52]:


os.chdir("D:\\TrainedModels")


# In[53]:


os.chdir("D:\\TrainedModels\\20221012")
model.save('AeroCNN-II_with_U-net_FlowFieldPrediction_50kernel_3x3_2by2MaxPooling_2CNNlayer_1block_testSize0.05.h5',
           overwrite=True, include_optimizer=True, save_format='h5')


# In[54]:


l2_error_train = np.sqrt(np.sum((decoded_train - y_train)**2) / np.sum(y_train**2))
print(l2_error_train)


# In[55]:


l2_error_test = np.sqrt(np.sum((decoded_test - y_test)**2) / np.sum(y_test**2))
print(l2_error_test)

