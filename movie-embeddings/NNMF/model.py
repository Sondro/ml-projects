from datetime import datetime
import multiprocessing
import time

import tensorflow as tf

from data_wrapper import DataReader


batch_size = 16
rank = 100
lambda_reg = .001
num_training_steps = 10**8
num_threads = multiprocessing.cpu_count()
start_time = datetime.now().strftime('%m-%d-%H-%M-%S')

graph = tf.Graph()
session = tf.Session(
    graph=graph,
    config=tf.ConfigProto(
        intra_op_parallelism_threads=num_threads
    )
)

reader = DataReader(
    use_tfrecords=False,
    num_threads=max(num_threads, 4),
    buffer_size=10**6
)
num_users = len(reader.user_index)
num_movies = len(reader.movie_index)

with graph.as_default():

    if reader.use_tfrecords:
        i, j, V_ij = reader.generate_batch(batch_size)
    else:
        i = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        j = tf.placeholder(dtype=tf.int32, shape=[batch_size])
        V_ij = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])

    W = tf.Variable(tf.truncated_normal([num_users, rank]))
    H = tf.Variable(tf.truncated_normal([num_movies, rank]))
    W_bias = tf.Variable(tf.truncated_normal([num_users]))
    H_bias = tf.Variable(tf.truncated_normal([num_movies]))

    global_mean = tf.constant(3.0)
    w_i = tf.gather(W, i)
    h_j = tf.gather(H, j)

    # w_bias and h_bias separate from W and H to remain unregularized
    w_bias = tf.gather(W_bias, i)
    h_bias = tf.gather(H_bias, j)
    interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)
    preds = global_mean + w_bias + h_bias + interaction

    # calculate regularized loss and perform gradient update
    RMSE = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, V_ij)))
    l2_norm = tf.sqrt(tf.reduce_sum(tf.square(w_i) + tf.square(h_j)))
    regularized_loss = RMSE + (lambda_reg * l2_norm)
    step = tf.train.GradientDescentOptimizer(1).minimize(regularized_loss)

    saver = tf.train.Saver()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())


with session.as_default():
    session.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session, coord)

    average_loss = 0
    log_interval = 500
    last_time = time.time()

    warm_start_init_step = 0
    if warm_start_init_step != 0:
        ckpt_file = 'checkpoints/model_{}d-{}'.format(
            rank, warm_start_init_step)
        saver.restore(session, ckpt_file)

    if not reader.use_tfrecords:
        batch_generator = reader.generate_batch(batch_size)

    for step_num in range(warm_start_init_step, num_training_steps):

        kwargs = {}
        if not reader.use_tfrecords:
            i_, j_, V_ij_ = batch_generator.next()
            kwargs['feed_dict'] = {i: i_, j: j_, V_ij: V_ij_}

        _, batch_loss, y, y_ = session.run([step, RMSE, V_ij, preds], **kwargs)

        average_loss += batch_loss
        if step_num % log_interval == 0:
            print step_num,
            print average_loss / float(log_interval),
            print time.time() - last_time
            with open('logs/train_log-{}.txt'.format(start_time), 'a') as log:
                log.write('{} {}\n'.format(step_num, batch_loss))
            last_time = time.time()
            average_loss = 0

        if step_num % 10000 == 0 and step_num > 0:
            saver.save(session, 'checkpoints/model_{}d'.format(rank),
                       global_step=step_num)

    coord.request_stop()
    coord.join(threads)
