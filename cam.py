#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import keras
import os
import tensorflow as tf
import onnx
import onnxruntime 


# In[2]:


def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        frame = frame[start: start + w]
    else:
        frame = frame[:, start: start + h]
    return frame


# In[3]:


def main():
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXYZ ')
    mean = 0.485 * 255.
    std = 0.229 * 255.
    
    this_session = onnxruntime.InferenceSession("asl.onnx")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std
        x = x.reshape(-1, 28, 28, 1).astype(np.float32)
        y = this_session.run(None, {'conv2d_input': x})[0]
        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)
        cv2.imshow("ASL Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()




