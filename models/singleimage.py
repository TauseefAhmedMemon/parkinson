import cv2 
import tensorflow as tf 
import numpy as np
from tensorflow import keras
mod=keras.models.load_model("/home/hoola/code/MobileNetLiveliness/models")
img=cv2.imread("/home/hoola/code/MobileNetLiveliness/data/processed/dataset/Test/Spoof/513107.png")
size=(224,224)
test=cv2.resize(img,size)
dim=np.expand_dims(test, axis=0)
prediction=mod.predict(dim)

print(prediction) 
