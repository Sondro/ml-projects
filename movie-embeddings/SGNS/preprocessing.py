import os
import gzip
import cPickle as pickle
from collections import defaultdict
from datetime import date, timedelta as td
import re

import numpy as np
from scipy.sparse import coo_matrix


def save_matrix(matrix, fname):
    with gzip.open(fname, 'wb') as f:
        pickle.dump(matrix, f, protocol=2)


def load_matrix(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def get_nnz(dirname):
    nnz = 0
    for fname in os.listdir(dirname):
        f = open(dirname + fname, 'r')
        nnz += sum(1 for line in f) - 1
    return nnz


def make_date_index(save_file):
    if os.path.isfile(save_file):
        return pickle.load(open(save_file, 'r'))

    earliest = date(1998, 10, 1)
    latest = date(2006, 1, 1)
    delta = latest - earliest

    date_index = defaultdict(lambda: len(date_index) + 1)
    for i in range(delta.days + 1):
        date_index[str(earliest + td(days=i))]

    pickle.dump(date_index, open(save_file, 'w'))
    return date_index


def make_movie_index(read_dir, save_file):
    if os.path.isfile(save_file):
        return pickle.load(open(save_file, 'r'))

    with open(read_dir, 'r') as movie_file:
        movie_index = {}
        movie_index[0] = 'NONE'
        for i, line in enumerate(movie_file):
            items = line.strip().split(',')
            movie_id = int(items[0])
            movie_title = ','.join(items[2:])
            movie_index[movie_id] = movie_title

        pickle.dump(movie_index, open(save_file, 'w'))
        return movie_index


def truncate_movie_list(threshold, read_dir, write_dir, movie_index):
    fnames = os.listdir(read_dir)

    i = 0
    movie_index_trunc, counts = {}, {}
    for fname in fnames:

        num_views = 0
        with open(read_dir + fname, 'r') as f:
            movie = int(f.next().strip()[:-1])
            num_views = sum(1 for line in f)
            counts[movie_index[movie]] = num_views

        if num_views >= threshold:
            with open(read_dir + fname, 'r') as f:
                new_fname = re.sub('[0-9]+', str(i).zfill(7), fname)
                with open(write_dir + new_fname, 'w') as new_file:
                    for line_num, line in enumerate(f):
                        if line_num == 0:
                            movie_id = int(line[:-2])
                            movie_index_trunc[i] = movie_index[movie_id]
                            line = str(i) + ':\n'
                            new_file.write(line)
                            i += 1
                        else:
                            new_file.write(line)

    pickle.dump(movie_index_trunc, open('cache/movie_index.pkl', 'w'))


def create_matrices(read_dir, date_index):
    nnz = get_nnz(read_dir)
    users, movies, ratings, dates = [np.zeros(nnz) for i in range(4)]
    fnames = [read_dir + fname for fname in os.listdir(read_dir)]
    user_index = defaultdict(lambda: len(user_index))

    i = 0
    for fname in fnames:
        with open(fname, 'r') as f:
            movie = int(f.next().strip()[:-1])
            j = 0
            for line in f:
                items = line.split(',')
                users[i] = user_index[items[0]]
                ratings[i] = items[1]
                dates[i] = date_index[items[-1].strip()]
                i, j = i + 1, j + 1
            movies[i - j: i] = movie

    rating_matrix = coo_matrix((ratings, (users, movies)), dtype=np.uint8)
    date_matrix = coo_matrix((dates, (users, movies)), dtype=np.uint16)
    rating_matrix = rating_matrix.tocsr()
    date_matrix = date_matrix.tocsr()
    save_matrix(rating_matrix, 'data/data_matrices/rating_matrix.pkl.gzip')
    save_matrix(date_matrix, 'data/data_matrices/date_matrix.pkl.gzip')
    pickle.dump(dict(user_index), open('data/data_matrices/user_index.pkl', 'w'))


def make_movie_dist(read_dir, save_file):
    if os.path.isfile(save_file):
        return pickle.load(open(save_file, 'r'))

    fnames = [read_dir + fname for fname in os.listdir(read_dir)]
    num_views = [sum(1 for line in open(fname, 'r')) - 1 for fname in fnames]
    pickle.dump(num_views, open(save_file, 'w'))


if __name__ == '__main__':
    raw_train_dir = 'data/train_raw/'
    truncated_train_dir = 'data/train_truncated/'

    date_index = make_date_index('cache/date_index.pkl')
    movie_index = make_movie_index(
        read_dir='data/movie_titles.txt',
        save_file='cache/movie_index.pkl'
    )
    truncate_movie_list(1000, raw_train_dir, truncated_train_dir, movie_index)
    make_movie_dist(
        read_dir='data/train_raw/',
        save_file='cache/movie_dist.pkl'
    )
    create_matrices('data/train_truncated/', date_index)
