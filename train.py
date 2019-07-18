from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.contrib import signal

import models
import Preprocess

def dataloader(path, process_fn=None):
    def parse_fn(example):
        feature = {'wave' : tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)}
        return tf.parse_single_example(example, feature)['wave']
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_fn, num_parallel_calls=4)
    if process_fn:
        dataset = dataset.map(process_fn, num_parallel_calls=4)
    #dataset = dataset.batch(1)
    dataset = dataset.repeat(-1)
    next = dataset.make_one_shot_iterator().get_next() 
    return next

class preproc:
    def __init__(self, length=256*30,  **kwargs):
        self.length = length
        self.onehot_depth = kwargs['Mu_law']['mu']
        self.mulaw = Preprocess.MuLaw(**kwargs['Mu_law'])
        self.spectorogram = Preprocess.Spectorogram(**kwargs['spec'])
    
    def preprocess(self, example):
        step = tf.cast(tf.random.uniform([], minval=0, maxval=self.length, dtype=tf.float32), tf.int32)
        x = signal.frame(example, self.length, step, pad_end=True)
        quatized = self.mulaw.transform(x)
        onehot = tf.one_hot(quatized, self.onehot_depth, 1.0, 0.0)
        normalized = tf.div(tf.cast(quatized, tf.float32), float(self.mulaw.mu), name='normalize')
        spec = self.spectorogram.make_mel_spectorogram(x)
        spec = self.spectorogram.power2db(spec)
        return quatized, onehot, normalized, spec
    
    def reverse(self, x):
        return self.mulaw.reverse(x)

def parseargs():
    parser = ArgumentParser()
    #MuLaw paramaters
    parser.add_argument('-m', '--mu', default=256)

    #spectorogram parmaters
    parser.add_argument('-r', '--sr', default=16000)
    parser.add_argument('--nfft', default=1024)
    parser.add_argument('-hop', '--hop_length', default=256)

    #other paramaters
    parser.add_argument('--crop', default=7680)

    #Wavenet paramaters
    parser.add_argument('--res_layer', default=10)
    parser.add_argument('--loop', default=2)
    parser.add_argument('--k_size', default=2)
    parser.add_argument('--res_channel', default=64)
    parser.add_argument('--skip_channel', default=256)

    args = parser.parse_args()
    return args

def train(args):
    spec_param = {'sampling_rate' : args.sr, 'nfft' : args.nfft, 'hop_length' : args.hop_length}
    ml_param = {'mu' : args.mu, 'int_type' : tf.int32, 'float_type' : tf.float32}
    preprocess_ = preproc(length=args.crop, **{'Mu_law' : ml_param, 'spec' : spec_param})
    quatized, onehot, normalized, spec = dataloader('data/train_arlu.tfrecord', process_fn=preprocess_.preprocess)
    Wavenet = models.Wavenet(args.loop, args.res_layer, args.k_size, args.res_channel, args.skip_channel, args.mu)
    Upsample = models.UpsampleNet(args.loop*args.res_layer, args.res_channel*2)
    enc_features = Upsample(spec)
    output = Wavenet(onehot, enc_features)
    loss = tf.losses.softmax_cross_entropy(onehot, output)
    tf.summary.scalar('loss', loss)
    opt = tf.train.AdamOptimizer(0.0001).minimize(loss)
    with tf.Session() as sess:
        loss_, _ = sess.run([loss, opt])

if __name__ == "__main__":
    args = parseargs()
    train(args)