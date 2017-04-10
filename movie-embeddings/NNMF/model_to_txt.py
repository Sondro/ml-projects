import tensorflow as tf
from data_wrapper import DataReader


reader = DataReader()
rank = 50

num_users = len(reader.user_index)
num_movies = len(reader.movie_index)


with tf.Graph().as_default():
    W = tf.Variable(tf.truncated_normal([num_users, rank]))
    H = tf.Variable(tf.truncated_normal([num_movies, rank]))
    # W_bias = tf.Variable(tf.truncated_normal([num_users]))
    # H_bias = tf.Variable(tf.truncated_normal([num_movies]))
    norm = tf.sqrt(tf.reduce_sum(tf.square(H), 1, keep_dims=True))
    normalized_embeddings = H / norm
    saver = tf.train.Saver()

    with tf.Session() as session:
        ckpt_file = 'checkpoints/model-{}'.format(26500000)

        saver.restore(session, ckpt_file)

        embeddings = session.run(normalized_embeddings)
        index = reader.movie_index
        movies = [movie for encoding, movie in sorted(index.items())]
        with open('embeddings/embeddings_50d.txt', 'w') as file:
            for movie, embedding in zip(movies, embeddings):
                movie_str = '_'.join(movie.split())
                file.write(movie_str + ' ')
                embed_str =  ' '.join(['{0:.8f}'.format(i) for i in embedding])
                file.write(embed_str + '\n')
