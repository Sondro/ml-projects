from preprocessing import load_matrix
import numpy as np
import tensorflow as tf
import cPickle as pickle


def make_example(rating, target_rating, movie, target_movie):
    context = [rating, target_rating, movie]
    context = [int(i) for i in context]
    label = [target_movie]
    label = [int(i) for i in label]
    features = {
        'context': tf.train.Feature(
            int64_list=tf.train.Int64List(value=context)),
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=label))
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    return ex


def write_data_files(
    skip_window,
    num_skips,
    write_text_files,
    write_tfrecord_files
):
    ratings_matrix = load_matrix('data/data_matrices/rating_matrix.pkl.gzip')
    dates_matrix = load_matrix('data/data_matrices/date_matrix.pkl.gzip')

    if write_text_files:
        writer = open('data/skipgrams.txt', 'w')
    elif write_tfrecord_files:
        writer = tf.python_io.TFRecordWriter('data/skipgrams.tfrecords')

    with writer:
        for i in range(ratings_matrix.shape[0]):
            ratings = np.array(ratings_matrix[i].todense()).ravel()
            dates = np.array(dates_matrix[i].todense()).ravel()

            movies = np.argsort(dates)
            ratings = ratings[movies]

            ratings = ratings[ratings != 0]
            movies = movies[-len(ratings):]

            for mid in range(skip_window, len(movies) - skip_window - 1):
                target = mid
                invalid = [mid]
                for j in range(num_skips):
                    while target in invalid:
                        window = (-skip_window, skip_window + 1)
                        offset = np.random.randint(*window)
                        target = mid + offset
                    invalid.append(target)

                    vals = [
                        ratings[mid],
                        ratings[target],
                        movies[mid],
                        movies[target]
                    ]
                    if write_text_files:
                        writer.write(' '.join([str(i) for i in vals]) + '\n')
                    elif write_tfrecord_files:
                        ex = make_example(*vals)
                        serialized = ex.SerializeToString()
                        writer.write(serialized)


def shard_text_file(full_file, shard_size):
    shard_num = 0
    current_fill = 0

    file_fmt = 'data/shards/shard-{}.txt'
    write_file = open(file_fmt.format(shard_num), 'w')
    for line in open(full_file, 'r'):
        write_file.write(line)
        current_fill += 1
        if current_fill == shard_size:
            current_fill = 0
            shard_num += 1
            write_file.close()
            write_file = open(file_fmt.format(shard_num), 'w')
    write_file.close()


def check_reconstruction(num_samples):
    user_index = pickle.load(open('cache/user_index.pkl', 'r'))
    date_index = pickle.load(open('cache/date_index.pkl', 'r'))

    date_index = dict(map(reversed, date_index.items()))
    user_index = dict(map(reversed, user_index.items()))

    ratings_matrix = load_matrix('data/data_matrices/rating_matrix.pkl.gzip')
    dates_matrix = load_matrix('data/data_matrices/date_matrix.pkl.gzip')

    for i in np.random.randint(0, ratings_matrix.shape[0], size=num_samples):
        ratings = np.array(ratings_matrix[i].todense()).ravel()
        dates = np.array(dates_matrix[i].todense()).ravel()

        movies = np.argsort(dates)
        ratings = ratings[movies]
        dates = dates[movies]

        ratings = ratings[ratings != 0]
        dates = dates[dates != 0]
        movies = movies[-len(ratings):]

        user = user_index[i]
        for movie, date, rating in zip(movies, dates, ratings):
            fname_id = '0'*(7 - len(str(movie))) + str(movie)
            fname = 'mv_' + fname_id + '.txt'

            date = date_index[date]
            with open('data/train_raw/' + fname, 'r') as movie_file:
                expected = str(user) + ',' + str(rating) + ',' + str(date)
                found_match = False
                for line in movie_file:
                    if expected == line.strip():
                        found_match = True
                        break
                assert found_match, \
                    "entry '{expected}' not found in {fname}".format(
                        expected=expected,
                        fname=fname
                    )
        assert np.all(sorted(dates) == dates)


# shard_text_file('data/skipgrams.txt', shard_size=5000)

# if __name__ == '__main__':
#     skip_window = 3
#     num_skips = 3
#     write_text_files = True
#     write_tfrecord_files = False
#     write_data_files(skip_window, num_skips, write_text_files,
#                      write_tfrecord_files)
#     check_reconstruction(100)
