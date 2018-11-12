"""Converts Adobe Triage data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageOps
import argparse
import random
from tqdm import tqdm
import io
import os
import sys
import tensorflow as tf

FLAGS = None
SIZE = (256, 256)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _resize(filename):
    image = Image.open(filename)
    image = image.resize(SIZE, Image.LANCZOS)
    bytes = io.BytesIO()
    image.save(bytes, format='JPEG', optimize=True)

    return bytes.getvalue()


def convert_to_split(image_directory, data, name):    
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    tf.gfile.MakeDirs(FLAGS.directory)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for item in tqdm(data):
            _ = int(item[0])
            image_id = int(item[1])
            rating1 = int(item[2])
            rating2 = int(item[3])
            rating3 = int(item[4])
            rating4 = int(item[5])
            rating5 = int(item[6])
            rating6 = int(item[7])
            rating7 = int(item[8])
            rating8 = int(item[9])
            rating9 = int(item[10])
            rating10 = int(item[11])

            image_filename = '%d.jpg' % (image_id)
            img = _resize(os.path.join(image_directory, image_filename))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img_jpg': _bytes_feature(img),
                        'ratings': _int64_feature([rating1, rating2, rating3, rating4, rating5, rating6, rating7, rating8, rating9, rating10]),
                    }
                )
            )
            writer.write(example.SerializeToString())


def convert_to(image_directory, imagelist):
    """Converts a dataset to tfrecords."""
    data = [line.strip('\n').split(' ') for line in open(imagelist)]
    random.shuffle(data)

    split_length = int(len(data) * 0.8)
    convert_to_split(image_directory, data[:split_length], 'train')
    convert_to_split(image_directory, data[split_length:], 'eval')


def main(unused_argv):
    base_directory = os.path.join(FLAGS.source)
    convert_to(os.path.join(base_directory, 'images'),
               os.path.join(base_directory, 'AVA.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/tmp/data',
        help='Directory to write the converted result'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='.',
        help='Directory to read raw data'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
