import tensorflow as tf
import numpy as np
import cPickle as pickle
import os


class DataReader(object):

    def __init__(self):
        self.movie_dist = pickle.load(open('cache/movie_dist.pkl', 'r'))
        self.movie_index = pickle.load(open('cache/movie_index.pkl', 'r'))

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

    def get_batch(self, batch_size):
        input_file = 'data/skipgrams.tfrecords'
        queue = tf.train.string_input_producer([input_file])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'context': tf.FixedLenFeature([3], tf.int64),
                'label': tf.FixedLenFeature([1], tf.int64)
            }
        )
        x, y = tf.train.batch(
            [features['context'], features['label']],
            batch_size=batch_size,
            num_threads=5
        )
        rating = x[:, 0]
        target_rating = x[:, 1]
        movie = x[:, 2]
        y = tf.reshape(y, [batch_size, 1])
        return rating, target_rating, movie, y

    def get_feed_batch(self, batch_size):
        buffer_size = 100000
        root_dir = 'data/shards/'
        fnames = np.array([root_dir + fname for fname in os.listdir(root_dir)])

        x_buffer, y_buffer = [], []
        while True:
            np.random.shuffle(fnames)
            for fname in fnames:
                with open(fname, 'r') as movie_file:
                    for line in movie_file:
                        row = map(int, line.split())
                        x_buffer.append(row[-2])
                        y_buffer.append(row[-1])

                        if len(x_buffer) < buffer_size:
                            continue

                        X = np.array(x_buffer)
                        Y = np.array(y_buffer)

                        indices = np.arange(buffer_size)
                        np.random.shuffle(indices)
                        prev = 0
                        for i in range(batch_size, buffer_size, batch_size):
                            idx = indices[prev: i]
                            yield (
                                X[idx],
                                Y[idx].reshape(-1, 1)
                            )
                            prev = i
                        x_buffer, y_buffer = [], []
