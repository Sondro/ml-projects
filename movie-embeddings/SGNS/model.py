from datetime import datetime
import tensorflow as tf
import numpy as np
import time

from data_wrapper import DataReader

reader = DataReader()
graph = tf.Graph()
session = tf.Session(
    graph=graph,
    config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
)

lr = .01
batch_size = 20
embedding_dim = 50
neg_samples = 10
training_steps = 10**8
vocab_size = len(reader.movie_index)
start_time = datetime.now().strftime('%m-%d-%H-%M-%S')

val_set = map(reader.encode_movie, [
    'Shrek 2',
    'Lord of the Rings: The Fellowship of the Ring',
    'The Matrix',
    'Men in Black',
    'Spider-Man 2',
    'Monsters, Inc.',
    'Good Will Hunting',
    'The School of Rock',
    'Napoleon Dynamite',
    'Mean Girls',
    'Dead Poets Society',
    'Star Wars: Episode II: Attack of the Clones',
    'Elf'
])

with graph.as_default():

    # read input data
    movies = tf.placeholder(dtype=tf.int32, shape=[batch_size])
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, 1])

    # model parameters
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0)
    )
    nce_weights = tf.Variable(
        tf.truncated_normal(
            shape=[vocab_size, embedding_dim],
            stddev=1.0 / np.sqrt(embedding_dim)
        )
    )
    nce_biases = tf.Variable(tf.zeros([vocab_size]))

    # feedforward and backpropogate with negative samples ~P(unigram) ^ (3/4)
    inputs = tf.nn.embedding_lookup(embeddings, movies)
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=tf.cast(labels, tf.int64),
        num_true=1,
        num_sampled=neg_samples,
        unique=True,
        range_max=vocab_size,
        distortion=0.75,
        unigrams=reader.movie_dist
    )
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, inputs, labels,
                       neg_samples, vocab_size, sampled_values=sampled_values)
    )
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # evaluate nearest neighbors of valdation data
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    validation_movies = tf.constant(val_set, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, validation_movies)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True
    )

    saver = tf.train.Saver()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())


with session.as_default():

    session.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    batches = reader.get_feed_batch(batch_size)

    average_loss = 0
    log_frequency = 1000
    last_time = time.time()

    warm_start_init_step = 0
    if warm_start_init_step != 0:
        ckpt_file = 'checkpoints/model_{}d-{}'.format(
            embedding_dim, warm_start_init_step)
        saver.restore(session, ckpt_file)

    for step_num in range(warm_start_init_step, training_steps):
        m, l = batches.next()
        _, batch_loss = session.run([optimizer, loss],
                                    feed_dict={movies: m, labels: l})
        average_loss += batch_loss
        if step_num % log_frequency == 0:
            average_batch_loss = average_loss / float(log_frequency)
            print step_num,
            print average_batch_loss
            print '({} sec)'.format(time.time() - last_time)
            with open('logs/train_log-{}.txt'.format(start_time), 'a') as log:
                log.write('{} {}\n'.format(step_num, average_batch_loss))
            last_time = time.time()
            average_loss = 0

        if step_num % 50000 == 0 and step_num > 0:
            sim = similarity.eval()
            for i in range(len(val_set)):
                top_k = 8
                target_movie = reader.decode_movie(val_set[i])
                nearest = (-sim[i, :]).argsort()[1: 8 + 1]
                print target_movie
                print '-' * len(target_movie)
                for k in range(top_k):
                    close_movie = reader.decode_movie(nearest[k])
                    print close_movie, sim[i, nearest[k]]
                print

        if step_num % 500000 == 0 and step_num > 0:
            saver.save(session, 'checkpoints/model_{}d'.format(embedding_dim),
                       global_step=step_num)

    coord.request_stop()
    coord.join(threads)
