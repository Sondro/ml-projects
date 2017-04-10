import cPickle as pickle
import os

import tensorflow as tf
import numpy as np


class DataReader(object):

    def __init__(self, use_tfrecords=False, num_threads=1, buffer_size=10*5):
        self.use_tfrecords = use_tfrecords
        self.num_threads = num_threads
        self.buffer_size = buffer_size
        self.user_index = pickle.load(open('cache/user_index.pkl', 'r'))
        self.movie_index = pickle.load(open('cache/movie_index.pkl', 'r'))
        self.read_dir = 'data/train_{}/'.format(
            'tfrecords' if use_tfrecords else 'txt')
        self.gen_fn = self.get_batch_as_tfrecords if use_tfrecords \
            else self.get_batch_for_feed_dict

    def decode_movie(self, index):
        return self.movie_index[index]

    def encode_movie(self, movie):
        if not hasattr(self, 'movie_index_inv'):
            self.movie_index_inv = dict(map(reversed, self.movie_index.items()))
        return self.movie_index_inv[movie]

    def decode_user(self, index):
        return self.user_index[index]

    def encode_user(self, user):
        if not hasattr(self, 'user_index_inv'):
            self.user_index_inv = dict(map(reversed, self.user_index.items()))
        return self.user_index_inv[user]

    def generate_batch(self, batch_size):
        return self.gen_fn(batch_size)

    def get_batch_as_tfrecords(self, batch_size):
        fnames = [self.read_dir + fname for fname in os.listdir(self.read_dir)]

        queue = tf.train.string_input_producer(fnames, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'w_i': tf.FixedLenFeature([1], tf.int64),
                'h_j': tf.FixedLenFeature([1], tf.int64),
                'V_ij': tf.FixedLenFeature([1], tf.int64)
            }
        )
        w_i, h_j, V_ij = tf.train.shuffle_batch(
            [features['w_i'], features['h_j'], features['V_ij']],
            batch_size=batch_size,
            num_threads=self.num_threads,
            capacity=self.buffer_size + batch_size * (self.num_threads + 2),
            min_after_dequeue=self.buffer_size
        )
        w_i = tf.squeeze(w_i)
        h_j = tf.squeeze(h_j)
        V_ij = tf.cast(tf.squeeze(V_ij), tf.float32)
        return w_i, h_j, V_ij

    def get_batch_for_feed_dict(self, batch_size):
        fnames = [self.read_dir + fname for fname in
                  os.listdir(self.read_dir) if not fname.startswith('.')]
        size = self.buffer_size
        while True:
            np.random.shuffle(fnames)
            data_buffer = []
            for fname in fnames:
                for line in open(fname, 'r'):
                    data_buffer.append(map(int, line.split(',')))
                    if len(data_buffer) == size:
                        data_buffer = np.array(data_buffer)
                        indices = np.arange(size)
                        np.random.shuffle(indices)

                        for i in range(0, size - batch_size, batch_size):
                            idx = indices[i: i + batch_size]
                            batch_data = data_buffer[idx]
                            yield (
                                batch_data[:, 0].ravel(),
                                batch_data[:, 1].ravel(),
                                batch_data[:, 2].reshape([-1, 1])
                            )
                        data_buffer = []
