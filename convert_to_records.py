"""Converts Adobe Triage data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageOps
import argparse
from tqdm import tqdm
import io
import os
import sys
import tensorflow as tf

FLAGS = None
SIZE = (224, 224)
PAD_COLOR = (128, 128, 128)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _resize_with_pad(filename):
    image = Image.open(filename)
    image.thumbnail(SIZE, Image.ANTIALIAS)
    background = Image.new('RGB', SIZE, PAD_COLOR)
    background.paste(
        image, (int((SIZE[0] - image.size[0]) / 2), int((SIZE[1] - image.size[1]) / 2))
    )
    bytes = io.BytesIO()
    background.save(bytes, format='JPEG', optimize=True)

    return bytes.getvalue()


def convert_to(image_directory, pairlist, name):
    """Converts a dataset to tfrecords."""
    pairs = [line.strip('\n').split(' ') for line in open(pairlist)]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    tf.gfile.MakeDirs(FLAGS.directory)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for pair in tqdm(pairs):
            series_id = int(pair[0])
            photo1_ind = int(pair[1])
            photo2_ind = int(pair[2])
            ratio = float(pair[3])

            photo1_filename = '%06d-%02d.JPG' % (series_id, photo1_ind)
            photo2_filename = '%06d-%02d.JPG' % (series_id, photo2_ind)

            img1 = Image.open(os.path.join(image_directory, photo1_filename))
            img2 = Image.open(os.path.join(image_directory, photo1_filename))

            img1 = _resize_with_pad(os.path.join(image_directory, photo1_filename))
            img2 = _resize_with_pad(os.path.join(image_directory, photo2_filename))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img1_raw': _bytes_feature(img1),
                        'img2_raw': _bytes_feature(img2),
                        'ratio_1>2': _float_feature(ratio),
                    }
                )
            )
            writer.write(example.SerializeToString())


def main(unused_argv):
    base_directory = os.path.join(FLAGS.source, 'train_val', 'train_val')
    convert_to(os.path.join(base_directory, 'train_val_imgs'),
               os.path.join(base_directory, 'val_pairlist.txt'),
               'validation')
    convert_to(os.path.join(base_directory, 'train_val_imgs'),
               os.path.join(base_directory, 'train_pairlist.txt'),
               'train')


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
