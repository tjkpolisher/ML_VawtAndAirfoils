#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AeroCNN-II
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import datetime


# In[ ]:


from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy(["/GPU:0","/GPU:1", "/GPU:2"], cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())


# In[2]:


n_kernel=100
l2Regularizer=1e-09
kernel_size1 = 5
kernel_size2 = 5
#kernel_size3 = 5
n_grid = 128
strides = 1
input_size = 100


# In[3]:


alpha = np.linspace(-10,20,16).reshape((16,1))


# In[4]:


aa = np.zeros((16*133,1))
for i in range(0, 133):
    aa[16*i:16*(i+1), :] = alpha[:,:]


# In[5]:


aa = aa.reshape((133, 16, 1, 1))


# In[6]:


os.chdir('D:\\AeroCNN2Inputs')


# In[7]:


origin = "D:\\AeroCNN2Inputs
origin_data = "D:\\AirfoilClCdCoordinates_out\\AirfoilClCdCoordinates_out"
origin_coord = "D:\\AirfoilClCdCoordinates_out\\AirfoilClCdCoordinates_out"


# In[8]:


folders_orig = os.listdir(origin)
folders = [file for file in folders_orig if file.endswith('.csv')]


# In[72]:


image_df = pd.read_csv('D:\\InputImages.csv', header=None)


# In[73]:


image_np = image_df.iloc[:, :].values


# In[75]:


image = image_np.reshape((133, 16, 100, 100))


# In[76]:


image = 1-image/100


# In[13]:


path = 'D:\\airfoilFlowField'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]


# In[14]:


file_name = origin_data + "\\AirfoilIndexList.xlsx"
airfoilName_df = pd.read_excel(file_name)
geometry_orig = airfoilName_df.iloc[:, 0].values


# In[15]:


data_name = path + '\\' + str(geometry_orig[0]) + "alpha"+ str(int(alpha[1])) + "_interpolated.csv"
print(data_name)


# In[16]:


path = 'D:\\rotatedInterpolation_pow2\\n_grid128\\velocityMagnitudeField'


# In[36]:


Vmag_df = pd.DataFrame()
for i in range(1, 134):
    for j in range(0, alpha.shape[0]):
        data_name = path + '\\' + str(geometry_orig[i-1]) + "alpha"+ str(int(alpha[j])) + "_velocityMagnitudeInterpolated.csv"
        data = pd.read_csv(data_name, header=None)
        Vmag_df = pd.concat([Vmag_df, data], axis=0)


# In[37]:


y_imp = Vmag_df.iloc[:, :].values


# In[38]:


y_imp.shape


# In[39]:


132*16*129


# In[40]:


(n_grid+1)**2 * 133 *16


# In[41]:


133*16


# In[43]:


y = y_imp.reshape((133, 16, n_grid+1, n_grid+1))


# In[44]:


geometry = np.zeros((133*16,1))
geometry = geometry.astype(np.string_)
for i in geometry_orig:
    index_ = np.where(geometry_orig==i)
    for j in range(0,16):
        geometry[16*index_[0]+j,:] = np.asarray(i)


# In[45]:


geometry.shape


# In[46]:


geometry = geometry.reshape((133, 16, 1))


# In[77]:


x_train, x_test, aa_train, aa_test, geo_train, geo_test, y_train, y_test = train_test_split(image, aa, geometry, y, test_size=0.05, random_state=1)


# In[78]:


x_train = x_train.reshape((x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))
x_test = x_test.reshape((x_test.shape[0]*x_test.shape[1], x_test.shape[2], x_test.shape[3], 1))
aa_train = aa_train.reshape((aa_train.shape[0]*aa_train.shape[1], aa_train.shape[2], aa_train.shape[3]))
aa_test = aa_test.reshape((aa_test.shape[0]*aa_test.shape[1], aa_test.shape[2], aa_test.shape[3]))
geo_train = geo_train.reshape((geo_train.shape[0]*geo_train.shape[1], geo_train.shape[2]))
geo_test = geo_test.reshape((geo_test.shape[0]*geo_test.shape[1], geo_test.shape[2]))
y_train = y_train.reshape((y_train.shape[0]*y_train.shape[1], y_train.shape[2], y_train.shape[3]))
y_test = y_test.reshape((y_test.shape[0]*y_test.shape[1], y_test.shape[2], y_test.shape[3]))


# In[68]:


with mirrored_strategy.scope():
    input_image = tf.keras.Input(shape=(100, 100,1))

    x_conv = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                    activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                    name='Conv2DLayer1')(input_image)
    x_ = tf.keras.layers.MaxPooling2D((2,2))(x_conv)
    x_conv = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                    activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                    name='Conv2DLayer2')(x_)
    x_ = tf.keras.layers.MaxPooling2D((2,2))(x_conv)
    x_conv = tf.keras.layers.Conv2D(n_kernel, (kernel_size1, kernel_size1), strides=(strides, strides),
                                    activation='relu', padding='same',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                    name='Conv2DLayer3')(x_)

    x_ = tf.keras.layers.Conv2DTranspose(n_kernel, (kernel_size1, kernel_size1), strides = (2, 2),
                                         padding='same', activation='relu')(x_)
    x_ = tf.keras.layers.Conv2DTranspose(n_kernel, (kernel_size1, kernel_size1), strides = (2, 2),
                                         padding='same', activation='relu')(x_)
    x_ = tf.keras.layers.Conv2DTranspose(n_kernel, (kernel_size1, kernel_size1), strides = (2, 2),
                                         padding='same', activation='relu')(x_)

    output_impr = tf.keras.layers.Conv2D(1, (kernel_size1, kernel_size1), strides=(strides, strides),
                                         activation='relu', padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2Regularizer),
                                         name='Conv2Doutput')(x_)
    x_ = tf.keras.layers.Flatten()(output_impr)
    x_ = tf.keras.layers.Dense(units=(n_grid+1)**2, activation='linear', name='outputLayer')(x_)
    output_data = tf.keras.layers.Reshape((n_grid+1, n_grid+1,1))(x_)
    # CNN autoencoder
    model = tf.keras.Model(input_image, output_data)


# In[69]:


with mirrored_strategy.scope():
    model.summary()


# In[81]:


with mirrored_strategy.scope():
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])


# In[82]:


with mirrored_strategy.scope():
    start = datetime.datetime.now()
    history = model.fit(x_train, y_train, epochs=5000, shuffle=True,
                        callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                                   restore_best_weights=True))
    end = datetime.datetime.now()


# In[ ]:


time = end - start
print("Training time:", time)


# In[ ]:


hist = history.history
plt.plot(hist['loss'], lw=2)
plt.title('Training loss (mean squared error)\nCNN Autoencoder', size=15)
plt.xlabel('Epoch', size=15)
plt.yscale('log')
#plt.ylim([5e-5, 1e-1])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()


# In[ ]:


plt.plot(hist['rmse'], lw=2)
plt.title('Root Mean Squared Error', size=15)
plt.xlabel('Epoch', size=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.show()


# In[ ]:


plt.plot(hist['rmse'], lw=2)
plt.title('Root Mean Squared Error\nCNN Autoencoder', size=15)
plt.xlabel('Epoch', size=15)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()


# In[ ]:


with mirrored_strategy.scope():
    test_results = model.evaluate(x_train)
    decoded_train = model.predict(x_train)
    decoded_test = model.predict(x_test)


# In[ ]:


test_results2 = model.evaluate(x_test)


# In[ ]:


y_train.shape


# In[ ]:


y_train.shape[0]


# In[ ]:


type(decoded_train.shape[3])


# In[ ]:


type(decoded_train)


# In[ ]:


type(n_grid)


# In[ ]:


decoded_train = decoded_train.reshape((int(y_train.shape[0]),int(n_grid),int(n_grid)))
decoded_test = decoded_test.reshape((int(y_test.shape[0]),int(n_grid),int(n_grid)))


# In[ ]:


decoded_train.shape


# In[ ]:


y_train.shape[0]


# In[ ]:


error_train_abs = np.abs(decoded_train - y_train)


# In[ ]:


error_test_abs = np.abs(decoded_test - y_test)


# In[ ]:


error_train_Cl_abs = np.abs(decoded_train[:,0,:] - y_train[:,0,:])


# In[ ]:


error_train_Cd_abs = np.abs(decoded_train[:,1,:] - y_train[:,1,:])


# In[ ]:


error_test_Cl_abs = np.abs(decoded_test[:,0,:] - y_test[:,0,:])


# In[ ]:


error_test_Cd_abs = np.abs(decoded_test[:,1,:] - y_test[:,1,:])


# In[ ]:


os.chdir("D:\\TrainedModels")


# In[ ]:


os.chdir("D:\\TrainedModels\\20221024")
model.save('CNNautoencoder_FlowFieldPrediction_100kernel_2by2MaxPooling_4blocks_testSize0.05.h5',
           overwrite=True, include_optimizer=True, save_format='h5')


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
print(l2_error_train_list)


# In[ ]:


l2_error_test_list = []
for i in range(0, len(aa_test)):
    l2_error_test_data = np.sqrt(np.sum((decoded_test[i] - y_test[i])**2) / np.sum(y_test[i]**2))
    l2_error_test_list.append(l2_error_test_data)
print(l2_error_test_list)


# In[ ]:


plt.plot(np.linspace(1, aa_train.shape[0], aa_train.shape[0]),
         l2_error_train*np.ones(aa_train.shape[0],), 'k', lw=2.5)
plt.scatter(np.linspace(1, aa_train.shape[0], aa_train.shape[0]), l2_error_train_list, c='b')
plt.xlabel('Index', fontsize=15)
plt.ylabel('$L_2$ error norm', fontsize=15)
plt.yscale('log')
plt.title('$L_2$ error norm variance - AeroCNN-II, training\nFlow field prediction, 1 CNN layer, 50 kernels', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


plt.plot(np.linspace(1, aa_test.shape[0], aa_test.shape[0]),
         l2_error_test*np.ones(aa_test.shape[0],), 'k', lw=2.5)
plt.scatter(np.linspace(1, aa_test.shape[0], aa_test.shape[0]), l2_error_test_list, c='b')
plt.xlabel('Index', fontsize=15)
plt.ylabel('$L_2$ error norm', fontsize=15)
plt.yscale('log')
plt.title('$L_2$ error norm variance - AeroCNN-II, test\nFlow field prediction, 1 CNN layer, 50 kernels', fontsize=15)
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


for c in range(0,16):
    plt.figure(figsize=(16, 8))
    y_test2_rotate = y_test[2*16+c].reshape(n_grid,n_grid)
    decoded_rotate = decoded_test[2*16+c].reshape(n_grid,n_grid)

    xrange = np.linspace(-2, 2, n_grid)
    yrange = np.linspace(-2, 2, n_grid)
    xmesh, ymesh = np.meshgrid(xrange, yrange)

    ax = plt.subplot(1, 2, 1)
    a1 = plt.contourf(xmesh, ymesh, y_test2_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Original test image', fontsize=15)

    # Display reconstruction
    ax = plt.subplot(1, 2, 2)
    a2 = plt.contourf(xmesh, ymesh, decoded_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Reconstructed image', fontsize=15)
    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(a1, cax=cax, orientation="horizontal")
    cbar.set_label('Velocity Magnitude', fontsize=15)
    #cbar.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.suptitle(r'Test dataset (%s, $\alpha$ = %d)' %(np.array2string(geo_test2[2*16+c])[3:-2], aa_test[2*16+c]),fontsize=20)
    plt.show()


# In[ ]:


for c in range(0,16):
    plt.figure(figsize=(16, 8))
    y_test0_rotate = y_test[c].reshape(n_grid,n_grid)
    decoded_rotate = decoded_test[c].reshape(n_grid,n_grid)

    xrange = np.linspace(-2, 2, n_grid)
    yrange = np.linspace(-2, 2, n_grid)
    xmesh, ymesh = np.meshgrid(xrange, yrange)

    ax = plt.subplot(1, 2, 1)
    a1 = plt.contourf(xmesh, ymesh, y_test0_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Original test image', fontsize=15)

    # Display reconstruction
    ax = plt.subplot(1, 2, 2)
    a2 = plt.contourf(xmesh, ymesh, decoded_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Reconstructed image', fontsize=15)
    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(a1, cax=cax, orientation="horizontal")
    cbar.set_label('Velocity Magnitude', fontsize=15)
    #cbar.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.suptitle(r'Test dataset (%s, $\alpha$ = %d)' %(np.array2string(geo_test2[c])[3:-2], aa_test[c]),fontsize=20)
    plt.show()


# In[ ]:


for c in range(0,16):
    plt.figure(figsize=(16, 8))
    y_train0_rotate = y_train[c].reshape(n_grid,n_grid)
    decoded_rotate = decoded_train[c].reshape(n_grid,n_grid)

    xrange = np.linspace(-2, 2, n_grid)
    yrange = np.linspace(-2, 2, n_grid)
    xmesh, ymesh = np.meshgrid(xrange, yrange)

    ax = plt.subplot(1, 2, 1)
    a1 = plt.contourf(xmesh, ymesh, y_train0_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Original test image', fontsize=15)

    # Display reconstruction
    ax = plt.subplot(1, 2, 2)
    a2 = plt.contourf(xmesh, ymesh, decoded_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Reconstructed image', fontsize=15)
    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(a1, cax=cax, orientation="horizontal")
    cbar.set_label('Velocity Magnitude', fontsize=15)
    #cbar.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.suptitle(r'Training dataset (%s, $\alpha$ = %d)' %(np.array2string(geo_train2[c])[3:-2], aa_train[c]),fontsize=20)
    plt.show()


# In[ ]:


for c in range(0,16):
    plt.figure(figsize=(16, 8))
    y_train20_rotate = y_train[20*16+c].reshape(n_grid,n_grid)
    decoded_rotate = decoded_train[20*16+c].reshape(n_grid,n_grid)

    xrange = np.linspace(-2, 2, n_grid)
    yrange = np.linspace(-2, 2, n_grid)
    xmesh, ymesh = np.meshgrid(xrange, yrange)

    ax = plt.subplot(1, 2, 1)
    a1 = plt.contourf(xmesh, ymesh, y_train20_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Original test image', fontsize=15)

    # Display reconstruction
    ax = plt.subplot(1, 2, 2)
    a2 = plt.contourf(xmesh, ymesh, decoded_rotate, vmin=0, vmax=11, levels=128, cmap='seismic')
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.set_title('Reconstructed image', fontsize=15)
    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(a1, cax=cax, orientation="horizontal")
    cbar.set_label('Velocity Magnitude', fontsize=15)
    #cbar.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    plt.suptitle(r'Training dataset (%s, $\alpha$ = %d)' %(np.array2string(geo_train2[20*16+c])[3:-2], aa_train[20*16+c]),fontsize=20)
    plt.show()


# In[ ]:


for c in range(0,16):
    error_test_abs2_rotate = error_test_abs[2*16+c].reshape(n_grid,n_grid)

    fig5 = plt.figure(figsize = (8, 8))
    ax5 = fig5.add_subplot(111)
    mappable = ax5.contourf(xmesh, ymesh, error_test_abs2_rotate, levels=128, cmap='seismic')
    ax5.set_title(r'Absolute error (%s, $\alpha$ = %d, $\epsilon$ = %.4f)' %(np.array2string(geo_test2[2*16+c])[3:-2],
                                                                       aa_test[2*16+c], l2_error_test_list[2*16+c]), fontsize=16)
    ax5.set_xlabel('$y$', fontsize=15)
    ax5.set_ylabel('$z$', fontsize=15)

    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label('Error', fontsize=15)

    plt.show()


# In[ ]:


for c in range(0,16):
    error_test_abs2_rotate = error_test_abs[c].reshape(n_grid,n_grid)

    fig5 = plt.figure(figsize = (8, 8))
    ax5 = fig5.add_subplot(111)
    mappable = ax5.contourf(xmesh, ymesh, error_test_abs2_rotate, levels=128, cmap='seismic')
    ax5.set_title(r'Absolute error (%s, $\alpha$ = %d, $\epsilon$ = %.4f)' %(np.array2string(geo_test2[c])[3:-2],
                                                                       aa_test[c], l2_error_test_list[c]), fontsize=16)
    ax5.set_xlabel('$y$', fontsize=15)
    ax5.set_ylabel('$z$', fontsize=15)

    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label('Error', fontsize=15)

    plt.show()


# In[ ]:


for c in range(0,16):
    error_train_abs2_rotate = error_train_abs[c].reshape(n_grid,n_grid)

    fig5 = plt.figure(figsize = (8, 8))
    ax5 = fig5.add_subplot(111)
    mappable = ax5.contourf(xmesh, ymesh, error_train_abs2_rotate, levels=128, cmap='seismic')
    ax5.set_title(r'Absolute error (%s, $\alpha$ = %d, $\epsilon$ = %.4f)' %(np.array2string(geo_train2[c])[3:-2],
                                                                       aa_train[c], l2_error_train_list[c]), fontsize=16)
    ax5.set_xlabel('$y$', fontsize=15)
    ax5.set_ylabel('$z$', fontsize=15)

    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label('Error', fontsize=15)

    plt.show()


# In[ ]:


for c in range(0,16):
    error_train_abs2_rotate = error_train_abs[16*20+c].reshape(n_grid,n_grid)

    fig5 = plt.figure(figsize = (8, 8))
    ax5 = fig5.add_subplot(111)
    mappable = ax5.contourf(xmesh, ymesh, error_train_abs2_rotate, levels=128, cmap='seismic')
    ax5.set_title(r'Absolute error (%s, $\alpha$ = %d, $\epsilon$ = %.4f)' %(np.array2string(geo_train2[16*20+c])[3:-2],
                                                                       aa_train[16*20+c], l2_error_train_list[16*20+c]), fontsize=16)
    ax5.set_xlabel('$y$', fontsize=15)
    ax5.set_ylabel('$z$', fontsize=15)

    cax = plt.axes([0.12, 0.005, 0.78, 0.05])
    cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label('Error', fontsize=15)

    plt.show()


# In[ ]:


file_name = origin_data + "\\AirfoilIndexList.xlsx"
airfoilName_df = pd.read_excel(file_name)
geometry_orig2 = airfoilName_df.iloc[:, 1].values


# In[ ]:


geometry = geometry.reshape((133, 16, 1))
geometry2 = np.zeros((133*16,1))
geometry2 = geometry2.astype(np.string_)
for i in geometry_orig2:
    index_ = np.where(geometry_orig2==i)
    for j in range(0,16):
        geometry2[16*index_[0]+j,:] = np.asarray(i)


# In[ ]:


geometry2 = geometry2.reshape((133, 16, 1))


# In[ ]:


x_train, x_test, geo_train, geo_test, geo_train2, geo_test2 = train_test_split(1-image_np.reshape((133, 16, 100, 100))/1000, geometry, geometry2, test_size=0.05, random_state=1)


# In[ ]:


geo_train2 = geo_train2.reshape((geo_train.shape[0]*geo_train.shape[1], geo_train.shape[2]))
geo_test2 = geo_test2.reshape((geo_test.shape[0]*geo_test.shape[1], geo_test.shape[2]))


# In[ ]:




