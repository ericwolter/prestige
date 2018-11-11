# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.INFO)

SIZE = 224

# defining input shape according to mobilenetV2
input_shape = (SIZE, SIZE, 3)
input1 = keras.layers.Input(input_shape)
input2 = keras.layers.Input(input_shape)

# first stage - siamese network with difference
m = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    
feature1 = m(input1)
feature2 = m(input2)
x = keras.layers.Subtract()([feature1, feature2])

# second stage - 2-layer perceptron with 2-way softmax
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(128, activation=tf.nn.tanh)(x)
x = keras.layers.Dense(128, activation=tf.nn.tanh)(x)
x = keras.layers.Dense(2, activation=tf.nn.softmax)(x)

model = keras.Model(inputs=[input1, input2], outputs=x)

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00001),
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

        img1 = tf.image.decode_jpeg(parsed_features['img1_raw'])
        img1 = tf.image.random_flip_left_right(img1)
        img1 = tf.to_float(img1)    
        img1 = tf.divide(img1, 128.)
        img1 = tf.subtract(img1, 1.)

        img2 = tf.image.decode_jpeg(parsed_features['img2_raw'])
        img2 = tf.image.random_flip_left_right(img2)
        img2 = tf.to_float(img2)    
        img2 = tf.divide(img2, 128.)
        img2 = tf.subtract(img2, 1.)

        ratio = parsed_features['ratio_1>2']

        features = {model.input_names[0]: img1,
                    model.input_names[1]: img2}

        def f1(): return [tf.constant(0), tf.constant(1)]
        def f2(): return [tf.constant(1), tf.constant(0)]
        labels = tf.cond(ratio < tf.constant(0.5), f1, f2)

        return features, labels

    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1000))  
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(8)
    dataset = dataset.prefetch(buffer_size=8)
    return dataset


def eval_input_fn():
    filenames = ["data/eval.tfrecords"]
    dataset = tf.data.TFRecordDataset(filenames)

    def _parse_function(example):
        features = {'img1_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'img2_raw': tf.FixedLenFeature((), tf.string, default_value=""),
                    'ratio_1>2': tf.FixedLenFeature((), tf.float32, default_value=0)}
        parsed_features = tf.parse_single_example(example, features)

        img1 = tf.image.decode_jpeg(parsed_features['img1_raw'])
        img1 = tf.to_float(img1)    
        img1 = tf.divide(img1, 128.)
        img1 = tf.subtract(img1, 1.)

        img2 = tf.image.decode_jpeg(parsed_features['img2_raw'])
        img2 = tf.to_float(img2)    
        img2 = tf.divide(img2, 128.)
        img2 = tf.subtract(img2, 1.)

        ratio = parsed_features['ratio_1>2']

        features = {model.input_names[0]: img1,
                    model.input_names[1]: img2}

        def f1(): return [tf.constant(0), tf.constant(1)]
        def f2(): return [tf.constant(1), tf.constant(0)]
        labels = tf.cond(ratio < tf.constant(0.5), f1, f2)

        return features, labels

    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(8)

    return dataset

for _ in range(10000):
    print('train')
    estimator.train(train_input_fn, steps=1000)
    print('eval')
    estimator.evaluate(eval_input_fn)
