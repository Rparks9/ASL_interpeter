#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras import models 
import os
import tensorflow as tf
import sys
import os
import cv2
from PIL import Image


def main():
    model = models.load_model('asl_model')
    letter_list = list('ABCDEFGHIJKLMNOPQRSTUWXYZ')
    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((64,64))
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        prediction  = np.argmax(model.predict(img_array), axis=-1)
        letter = letter_list[int(prediction)]
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("ASL Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
if __name__ == '__main__':
    main()




