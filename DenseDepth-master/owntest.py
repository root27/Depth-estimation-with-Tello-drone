import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, load_single_image, single_display_image
from matplotlib import pyplot as plt
import cv2

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model('nyu.h5', custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format('nyu.h5'))

# Input images

inputs = load_single_image('example2/IMG_4913.png')
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
'''
inputs = cv2.imread('examples/1_image.png')
'''
# Compute results
outputs = predict(model, inputs)
print(outputs.shape)
print((outputs[0].shape[0], outputs[0].shape[1], 3))
print(outputs.shape[0])
#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

#Display results
viz = display_images(outputs.copy())
print(viz.shape)
print(viz[:120,180:])
plt.figure(figsize=(10,5))
plt.imshow(viz)
#plt.savefig('test.png')
plt.show()