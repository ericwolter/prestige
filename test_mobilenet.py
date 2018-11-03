# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np

img_path = 'data/elephant.jpg'
img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = keras.preprocessing.image.img_to_array(img)
x = keras.applications.mobilenet_v2.preprocess_input(x)
print(x)

tf.enable_eager_execution()

with open('data/elephant.jpg', mode='rb') as file:
    value = file.read()

t = tf.image.decode_jpeg(value)
t = tf.image.convert_image_dtype(t, dtype=tf.float32)
t = (t - 0.5) * 2
print(t)

model = keras.applications.MobileNetV2(weights='imagenet')

preds = model.predict(t)
print('Predicted:', keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0])


