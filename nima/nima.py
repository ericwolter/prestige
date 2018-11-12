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

estimator = tf.keras.estimator.model_to_estimator(keras_model=nima, model_dir='model')

def train_input_fn():
    filenames = ["data/AVA/train.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(example):
        features = {'img_jpg': tf.FixedLenFeature((), tf.string, default_value=""),
                    'ratings': tf.FixedLenFeature((), tf.int64, default_value=[0,0,0,0,0,0,0,0,0])}
        parsed_features = tf.parse_single_example(example, features)

        feature = tf.image.decode_jpeg(parsed_features['img_jpg'])
        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_crop(feature, (224, 224))

        #preprocess for mobilenet
        feature = tf.to_float(feature)    
        feature = tf.divide(feature, 128.)
        feature = tf.subtract(feature, 1.)        

        labels = parsed_features['ratings']

        return feature, labels

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))  
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(8)
    dataset = dataset.prefetch(buffer_size=8)
    return dataset


def eval_input_fn():
    filenames = ["/content/gdrive/My Drive/prestige/data/eval.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(example):
        features = {'img_jpg': tf.FixedLenFeature((), tf.string, default_value=""),
                    'ratings': tf.FixedLenFeature((), tf.int64, default_value=[0,0,0,0,0,0,0,0,0])}
        parsed_features = tf.parse_single_example(example, features)

        feature = tf.image.decode_jpeg(parsed_features['img_jpg'])
        feature = tf.image.random_flip_left_right(feature)
        feature = tf.image.random_crop(feature, (224, 224))

        #preprocess for mobilenet
        feature = tf.to_float(feature)    
        feature = tf.divide(feature, 128.)
        feature = tf.subtract(feature, 1.)

        labels = parsed_features['ratings']

        return feature, labels

    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(8)

    return dataset

for _ in range(10000):
    print('train')
    estimator.train(train_input_fn, steps=10000)
    print('eval')
    estimator.evaluate(eval_input_fn)

