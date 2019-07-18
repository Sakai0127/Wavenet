import argparse
import os

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def make_example(data):
    return tf.train.Example(features=tf.train.Features(feature={
        'wave' : tf.train.Feature(float_list=tf.train.FloatList(value=data))
    }))

def write_data(args):
    datalist = os.listdir(args.datadir)
    if args.split_train_test:
        np.random.shuffle(datalist)
        n = int(len(datalist)*args.split_rate)
        with tf.python_io.TFRecordWriter('train_'+args.outputpath) as train_writer:
            for name in tqdm(datalist[:n]):
                path = os.path.join(args.datadir, name)
                w, _ = librosa.load(path, sr=args.samplingrate)
                train_writer.write(make_example(w).SerializeToString())
        with tf.python_io.TFRecordWriter('test_'+args.outputpath) as test_writer:
            for name in tqdm(datalist[n:]):
                path = os.path.join(args.datadir, name)
                w, _ = librosa.load(path, sr=args.samplingrate)
                test_writer.write(make_example(w).SerializeToString())
    else:
        with tf.python_io.TFRecordWriter(args.outputpath) as writer:
            for name in datalist:
                path = os.path.join(args.datadir, name)
                w, _ = librosa.load(path, sr=args.samplingrate)
                writer.write(make_example(w).SerializeToString())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=str)
    parser.add_argument('outputpath', type=str)
    parser.add_argument('-sr', '--samplingrate', type=int, default=16000)
    parser.add_argument('-d', '--split_train_test', action='store_true')
    parser.add_argument('-r', '--split_rate', type=float, default=0.9)

    args = parser.parse_args()
    write_data(args)