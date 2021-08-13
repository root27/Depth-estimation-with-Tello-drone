import os
import glob
import argparse
import matplotlib
import numpy as np
# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, load_single_image, single_display_image
from matplotlib import pyplot as plt
import cv2
from math import *
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format('nyu.h5'))
left_threshold = -40
right_threshold = 40
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(640,480))
    # Input images

    x = np.clip(np.asarray(rgb, dtype=float) / 255, 0, 1)
    inputs =  np.stack(x, axis=0)


    # Compute results
    outputs = predict(model, inputs)
    #print(outputs.shape)
    #print((outputs[0].shape[0], outputs[0].shape[1], 3))
    #print(outputs.shape[0])
    #matplotlib problem on ubuntu terminal fix
    #matplotlib.use('TkAgg')   

#   Display results
    viz = display_images(outputs.copy())
    viz2 = viz.astype('float32')
   
    gray_img = cv2.cvtColor(viz2, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(gray_img,(11,11),0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(frame)
    cv2.circle(frame, maxLoc, 11, (255, 0, 0), 2)
    print(maxLoc)
    diff = maxLoc[0] - 160
    if left_threshold > diff :
        calc = (2*abs(diff)*tan(radians(81/2))) / 320
        phi = degrees(atan(calc))
        print('Turning left with {}'.format(phi))
        
    elif right_threshold < diff:
        calc = (2*abs(diff)*tan(radians(81/2))) / 320
        phi = degrees(atan(calc))
        print('Turning right with {}'.format(phi))
    
    #Reference line
    cv2.line(frame,(160,0),(160,240),(0,255,0),1)


    cv2.imshow("depth", frame)
#plt.figure(figsize=(10,5))
#plt.imshow(viz)
#plt.savefig('test.png')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#plt.show()