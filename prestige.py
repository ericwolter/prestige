# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras

SIZE = 224

# defining input shape according to mobilenetV2
input_shape = (SIZE, SIZE, 3)
input1 = keras.layers.Input(input_shape)
input2 = keras.layers.Input(input_shape)

# first stage - siamese network with difference
m = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
feature1 = m(input1)
feature2 = m(input2)
subtract = keras.layers.Subtract()([feature1, feature2])

# second stage - 2-layer perceptron with 2-way softmax
out = keras.layers.Dense(128, activation=tf.nn.tanh)(subtract)
out = keras.layers.Dense(128, activation=tf.nn.tanh)(out)
out = keras.layers.Dense(2, activation=tf.nn.softmax)(out)

model = keras.Model(inputs=[input1, input2], outputs=out)
for layer in m.layers:
    layer.trainable = False

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='model')


def train_input_fn():
    filenames = ["data/train.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(example):
        features = {'img1_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'img2_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'ratio_1>2': tf.FixedLenFeature((), tf.float32, default_value=0)}
        parsed_features = tf.parse_single_example(example, features)

        img1_decoded = tf.image.decode_jpeg(parsed_features['img1_raw'], channels=3)
        img2_decoded = tf.image.decode_jpeg(parsed_features['img2_raw'], channels=3)
        ratio = parsed_features['ratio_1>2']

        features = {'input_1': img1_decoded,
                    'input_2': img2_decoded}
        labels = [ratio, 1 - ratio]

        return features, labels

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(4)
    return dataset


def eval_input_fn():
    filenames = ["data/eval.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(example):
        features = {'img1_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'img2_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'ratio_1>2': tf.FixedLenFeature((), tf.float32, default_value=0)}
        parsed_features = tf.parse_single_example(example, features)

        img1_decoded = tf.image.decode_jpeg(parsed_features['img1_raw'], channels=3)
        img2_decoded = tf.image.decode_jpeg(parsed_features['img2_raw'], channels=3)
        ratio = parsed_features['ratio_1>2']

        features = {'input_1': img1_decoded,
                    'input_2': img2_decoded}
        labels = [ratio, 1 - ratio]

        return features, labels

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(4)
    return dataset


estimator.train(train_input_fn, steps=2)
estimator.evaluate(eval_input_fn)
