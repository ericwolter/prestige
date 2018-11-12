# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

m = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet', pooling='avg')

# add dropout and dense layer
x = keras.layers.Dropout(0.75)(m.output)
x = keras.layers.Dense(units=10, activation='softmax')(x)

nima = keras.Model(m.inputs, x)
nima.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])

nima.summary()