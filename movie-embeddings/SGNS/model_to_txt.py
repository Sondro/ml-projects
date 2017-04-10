import tensorflow as tf
from data_wrapper import DataReader


reader = DataReader()
vocab_size = len(reader.movie_index)
embedding_dim = 100


with tf.Graph().as_default():
    embeddings = tf.Variable(
        tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0)
    )
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    saver = tf.train.Saver()

    with tf.Session() as session:
        ckpt_file = 'checkpoints/model_100d-{}'.format(24000000)

        saver.restore(session, ckpt_file)

        embeddings = session.run(normalized_embeddings)
        index = reader.movie_index
        movies = [movie for encoding, movie in sorted(index.items())]
        with open('embeddings/embeddings_100d.txt', 'w') as file:
            for movie, embedding in zip(movies, embeddings):
                movie_str = '_'.join(movie.split())
                file.write(movie_str + ' ')
                embed_str = ' '.join(['{0:.8f}'.format(i) for i in embedding])
                file.write(embed_str + '\n')
