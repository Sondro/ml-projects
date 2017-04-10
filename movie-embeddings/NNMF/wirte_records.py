from collections import defaultdict
from multiprocessing import Pool, cpu_count
import cPickle as pickle
import numpy as np
import os

import tensorflow as tf


def make_example(w_i, h_j, V_ij):
    features = {
        'w_i': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[w_i])),
        'h_j': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[h_j])),
        'V_ij': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[V_ij]))
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex


def create_user_index(read_dir):
    save_file = 'cache/user_index.pkl'
    if os.path.isfile(save_file):
        return pickle.load(open(save_file, 'r'))
    else:
        fnames = os.listdir(read_dir)
        user_index = defaultdict(lambda: len(user_index))
        for fname in fnames:
            fname = os.path.join(read_dir, fname)
            with open(fname, 'r') as read_file:
                read_file.next()
                for line in read_file:
                    row = line.strip().split(',')
                    user_index[int(row[0])]
        user_index = dict(user_index)
        pickle.dump(user_index, open(save_file, 'w'))
        return user_index


def write_tfrecord(fnames):
    global user_index
    read_fname, write_fname = fnames
    if os.path.isfile(write_fname):
        return
    with tf.python_io.TFRecordWriter(write_fname) as writer, \
            open(read_fname, 'r') as read_file:
        movie_id = int(read_file.next()[:-2])
        for line in read_file:
            row = line.strip().split(',')
            rating = int(row[1])
            user = user_index[int(row[0])]
            ex = make_example(user, movie_id, rating)
            serialized = ex.SerializeToString()
            writer.write(serialized)


def write_txt(fnames):
    global user_index
    read_fname, write_fname = fnames
    if os.path.isfile(write_fname):
        return
    with open(write_fname, 'w') as write_file, \
            open(read_fname, 'r') as read_file:
        movie_id = read_file.next()[:-2]
        for line in read_file:
            row = line.strip().split(',')
            rating = row[1]
            user = str(user_index[int(row[0])])
            write_file.write(user + ',' + movie_id + ',' + rating + '\n')


if __name__ == '__main__':
    use_tfrecords = False
    mode = 'tfrecords' if use_tfrecords else 'txt'
    dir_fmt = 'data/train_{}/'

    read_dir = dir_fmt.format('raw')
    read_fnames = [read_dir + fname for fname in os.listdir(read_dir)]
    write_fnames = [fname.replace('raw', mode) for fname in read_fnames]

    write_dir = dir_fmt.format(mode)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    read_write_fnames = zip(read_fnames, write_fnames)
    user_index = create_user_index(read_dir)
    write_function = write_tfrecord if use_tfrecords else write_txt
    pool = Pool(processes=(cpu_count() - 1))
    pool.map(write_function, read_write_fnames)
