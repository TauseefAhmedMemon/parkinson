import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
#image_preparation

train_path= '/home/hoola/code/MobileNetLiveliness/data/processed/dataset/Train'
valid_path='/home/hoola/code/MobileNetLiveliness/data/processed/dataset/Valid'
test_path='/home/hoola/code/MobileNetLiveliness/data/processed/dataset/Test'
train_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), batch_size=96)
valid_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), batch_size=96)
test_batches= ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), batch_size=96, shuffle=False)
print(type(train_batches),type(test_batches),type(valid_batches))

#assert train_batches.n == 17858
#assert valid_batches.n == 5256
#assert test_batches.n == 2729
assert train_batches.num_classes == valid_batches.num_classes==test_batches.num_classes==2

#Modify Model
mobile= tf.keras.applications.MobileNetV2(weights='imagenet')
print(len(mobile.layers))
x=mobile.layers[-2].output
output=Dense(units=2, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=output)
#
for layer in model.layers[:-100]:
    layer.trainable = False

# train the model
#print("TRAIN BATCH>>>>>>>>>>", list(train_batches), valid_batches)
model.compile(optimizer=Adam(learning_rate=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=1)
model_path='/home/hoola/code/MobileNetLiveliness/models'
model.save('/home/hoola/code/MobileNetLiveliness/models')

tf.keras.models.save_model(model,model_path)
#test the model
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks,classes)

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Non-Normalized Confusion Matrix')
    print(cm)

test_labels= test_batches.classes
predictions= model.predict(x=test_batches, verbose=0)
cm= confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
test_batches.class_indices
cm_plot_labels=['Cnic-Laminated-Back','Cnic-Laminated-Front', 'Cnic-Original-Back', 'Cnic-Original-Front', 'Cnic-Photocopied-Back', 'Cnic-Photocopied-Front']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
