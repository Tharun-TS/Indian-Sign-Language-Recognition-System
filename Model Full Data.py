
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[1]:


# importing pandas and numpy
import pandas as pd
import numpy as np


# In[2]:


# fixing random seed as 2018 for reproducibility..!!
seed = 2018
np.random.seed(seed)


# In[3]:


# loading libraries needed..!!
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical


# In[5]:


# loading training file..!!
train = pd.read_csv('shuffled_fulldata.csv')


# In[6]:


# function to read image and resize it.!!
def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    return img


# In[7]:


# path for training images
TRAIN_PATH = 'C:\\Users\\Spectre\\Downloads\\Project\\Full Data\\'


# In[8]:


# loading training images as arrays to train_img numpy array
train_img = []

for img_path in tqdm(train['image_id'].values):
    train_img.append(read_img(TRAIN_PATH + img_path))


# In[9]:


# normalizing the data for easy computations.!!
data_feats = np.array(train_img, np.float32) / 255.


# In[10]:


# labels for the training data..!!
labels = train['class'].tolist()


# In[11]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)


# In[12]:


# make labels categorical via one-hot encoding...!
labels = to_categorical(labels)


# In[13]:


# looking at the shape of the data..!! 

data_feats.shape


# In[14]:


# making train - valid sets..!!

from sklearn.model_selection import train_test_split

train_feats, valid_feats, train_labels, valid_labels = train_test_split(data_feats, labels, test_size=0.33, random_state=seed)


# In[4]:


# necessary layers for making a CNN..!!

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# In[5]:


# defining the model.!!

from keras.layers import Input, Dense
from keras.models import Model

inputs = Input((64, 64, 3), name='Input')

x = Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', activation ='relu', name = 'Conv2D_1')(inputs)
x = Conv2D(filters = 32, kernel_size = (5,5),padding = 'same', activation ='relu', name = 'Conv2D_2')(x)
x = MaxPooling2D(pool_size=(2,2), name = 'MaxPool2D_1')(x)
x = Dropout(0.25, name='Dropout_1')(x)

x = Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu', name = 'Conv2D_3')(x)
x = Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', activation ='relu', name = 'Conv2D_4')(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2), name = 'MaxPool2D_2')(x)
x = Dropout(0.25, name = 'Dropout_2')(x)

x = Flatten(name='Flatten')(x)

x = Dense(512, activation = "relu", name='Dense_1')(x)
x = Dropout(0.5, name='Dense_Dropout_1')(x)
x = Dense(128, activation = "relu", name='Dense_2')(x)

predictions = Dense(36, activation = "softmax", name='Dense_Output')(x)

model = Model(inputs=inputs, outputs=predictions)


# In[2]:


# compiling model with categorical_crossentropy loss and adam optimizer.!!

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[7]:


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png')


# In[37]:


# fitting the model on train data, evaluting on validatioon data, with batch_size as 64 for 17 epochs..!!

history = model.fit(traihn_feats, train_labels, validation_data=(valid_feats, valid_labels), batch_size=128, verbose=2, epochs=10)


# In[45]:


# taking a look at plot.!! - It's clear that there is no overfitting.!!

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[58]:


data_feats[0].shape


# In[46]:


# taking a look at plot.!! - It's clear that there is no overfitting.!!

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[47]:


score = model.evaluate(valid_feats, valid_labels, verbose=0)
score[1] * 100


# In[48]:


# function to plot confusion matrix..!

import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[49]:


labels_from_vector = np.argmax(labels, axis=1)
labels_from_vector


# In[50]:


unique_labels = np.unique(labels_from_vector)
unique_labels


# In[51]:


original_labels = encoder.inverse_transform(unique_labels)
original_labels


# In[52]:


from sklearn.metrics import confusion_matrix
# Predict the values from the validation dataset
Y_pred = model.predict(valid_feats)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(valid_labels, axis = 1)  
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = original_labels) 


# In[53]:


model.save('full_data_model.h5')


# In[54]:


from sklearn.metrics import precision_score, accuracy_score, recall_score


# In[55]:


from keras.models import load_model

full_model = load_model('full_data_model.h5')


# In[56]:


valid_preds = np.argmax(full_model.predict(valid_feats), axis=1)

precision_score(np.argmax(valid_labels, axis=1), valid_preds, average='weighted')


# In[57]:


recall_score(np.argmax(valid_labels, axis=1), valid_preds, average='weighted')


# In[59]:


import os

PATH = 'Test Data\\'

for test_files in os.listdir(PATH):
    
    print(test_files)
    
    test_feats = read_img(PATH + test_files)
    
    test_feats = np.array(test_feats, np.float32) / 255.
    test_feats = np.expand_dims(test_feats, axis=0)
    
    preds = np.argmax(model.predict(test_feats), axis=1)
    
    print(encoder.inverse_transform(preds))
    

