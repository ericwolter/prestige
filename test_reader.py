# pylint: disable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from PIL import Image

tf.enable_eager_execution()

dataset = tf.data.TFRecordDataset(['data/eval.tfrecords'])
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
    img2 = tf.to_float(img1)    
    img2 = tf.divide(img1, 128.)
    img2 = tf.subtract(img1, 1.)

    ratio = parsed_features['ratio_1>2']

    features = {'input1': img1,
                'input2': img2}
    labels = [ratio, 1 - ratio]

    return features, labels

dataset = dataset.map(_parse_function)
iterator = dataset.make_one_shot_iterator() 

record1 = iterator.next()
print(record1)
# A1=record1[0]['input1'].numpy()
# B1=record1[0]['input2'].numpy()

# im = Image.fromarray(A1)
# im.save("data/eval1A.jpg")
# im = Image.fromarray(B1)
# im.save("data/eval1B.jpg")

# record2 = iterator.next()
# A2=record2[0]['input1'].numpy()
# B2=record2[0]['input2'].numpy()

# im = Image.fromarray(A2)
# im.save("data/eval2A.jpg")
# im = Image.fromarray(B2)
# im.save("data/eval2B.jpg")
