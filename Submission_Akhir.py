#!/usr/bin/env python
# coding: utf-8

# In[29]:


dataset = "dataku"


# In[30]:


#Count datasets
import os
print('total cane images :', len(os.listdir(dataset + '/cane')))
print('total gallina images :', len(os.listdir(dataset + '/gallina')))
print('total ragno images :', len(os.listdir(dataset + '/ragno')))


total = sum([len(os.listdir(dataset + '/cane')),
             len(os.listdir(dataset + '/gallina')),
             len(os.listdir(dataset + '/ragno'))])

print("total datasets : ", total)


# In[31]:


#check image datasets resolution

from PIL import Image
img_dir = dataset+"/cane"
file_list = []
for file in os.listdir(img_dir):
  file_path = img_dir+'/'+file
  file_list.append(file_path)
i = 1
for file_name in file_list[1:20]:
  img = Image.open(file_name).convert('RGBA')
  print(img.size)


# In[32]:


#Image Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) #validation 20%


# In[33]:


data_train = train_datagen.flow_from_directory(
    dataset,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical',
    subset='training') # set as training data
data_validation = train_datagen.flow_from_directory(
    dataset, 
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical',
    subset='validation') # set as validation data


# In[42]:


#Build Model
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation='relu', padding='same', input_shape=(150,150,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(3, activation='softmax'))


# In[43]:


#compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics = ['accuracy'])


# In[44]:


model.summary()


# In[45]:


#callbacks
import time

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


# In[46]:


cnn = model.fit(data_train,
                validation_data=data_validation,
                epochs=20,
                callbacks=[time_callback],
                verbose=1)


# In[47]:


print("Total Training time : ", sum(time_callback.times))


# In[48]:


#accuracy visualization

import matplotlib.pyplot as plt

acc = cnn.history['accuracy']
val_acc = cnn.history['val_accuracy']

epochs = range(len(acc))

fig, ax = plt.subplots(figsize=(20,8))
ax.plot(epochs, acc, 'b', label='Training Accuracy')
ax.plot(epochs, val_acc, 'r', label='Validation Accuracy')
ax.set_title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[49]:


#loss visualization

import matplotlib.pyplot as plt

loss = cnn.history['loss']
val_loss = cnn.history['val_loss']

epochs = range(len(loss))

fig, ax = plt.subplots(figsize=(20,8))
ax.plot(epochs, loss, 'g', label='Training Loss')
ax.plot(epochs, val_loss, 'y', label='Validation Loss')
ax.set_title('Training and Validation Loss')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[ ]:


# Convert & Save TF-Lite Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)


# In[ ]:




