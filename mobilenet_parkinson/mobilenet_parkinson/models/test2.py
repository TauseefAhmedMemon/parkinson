import cv2 
import tensorflow as tf 
import numpy as np
from tensorflow import keras
mod=tf.keras.models.load_model("/home/hoola/code/face-recognition-with-liveness-web-login/face_recognition_and_liveness/face_liveness_detection/models/liveness.model")
cap=cv2.VideoCapture(0)
size=(32,32)
while (cap.isOpened()):
    ret,frame=cap.read()
    dim=cv2.resize(frame, size)
    test=np.expand_dims(dim, axis=0)
    predictions=mod.predict(test)
    print(predictions)
    clas=np.argmax(predictions)
    print(clas)
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

