import tensorflow as tf
import numpy as np


class MDN(object):

    def __init__(
        self,
        num_mixture_components=8,
        num_training_steps=10**5,
        activation_function='tanh',  # 'tanh', 'sigmoid', or 'relu'
        optimizer='adam',  # 'adam', 'gd', or 'ada'
        learning_rate=.01,
        hidden_units=32,
        batch_size=64,
        verbose=False  # logs training loss to stdout if True
    ):
        self.num_mixture_components = num_mixture_components
        self.num_training_steps = num_training_steps
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, Y):
        input_units = X.shape[1]
        output_dim = Y.shape[1]
        output_units = self.num_mixture_components * (
            output_dim +  # d parameters for covariance matrix (diagonal only)
            output_dim +  # d parameters for mean vector
            1  # mixing coefficient
        )
        self.graph = self._build_graph(input_units, output_units)
        self.session = tf.Session(graph=self.graph)

        with self.session.as_default():

            self.session.run(self.init)

            average_loss = 0
            batch_generator = self._batch_generator(X, Y)
            for step_num in range(self.num_training_steps):
                sample_x, sample_y = batch_generator.next()
                feed_dict = {self.x: sample_x, self.y: sample_y}

                loss = self.session.run(self.loss, feed_dict=feed_dict)
                self.session.run(self.step, feed_dict=feed_dict)

                average_loss += loss
                if step_num % 1000 == 0:
                    if self.verbose:
                        print 'step_num: {}, \tloss: {}'.format(
                            step_num, average_loss)
                    average_loss = 0

    def predict(self, X):
        pis, sigmas, mus = self.session.run(
            [self.pis, self.sigmas, self.mus],
            feed_dict={self.x: X}
        )
        Y = []
        for pi, sigma, mu in zip(pis, sigmas, mus):
            sample = self._sample_mixture(pi, sigma, mu)
            Y.append(sample)
        return Y

    def _build_graph(self, input_units, output_units):
        graph = tf.Graph()

        with graph.as_default():

            self.x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, input_units]
            )
            self.y = tf.placeholder(
                dtype=tf.float32,
                shape=[None, input_units]
            )

            W_in = tf.get_variable(
                name='W_in',
                initializer=tf.random_normal_initializer(
                    mean=0.0,
                    stddev=(1.0 / np.sqrt(input_units))
                ),
                shape=[1, self.hidden_units]
            )
            b_in = tf.get_variable(
                name='b_in',
                initializer=tf.constant_initializer(0.0),
                shape=[self.hidden_units]
            )

            W_out = tf.get_variable(
                name='W_out',
                initializer=tf.random_normal_initializer(
                    mean=0.0,
                    stddev=(1.0 / np.sqrt(self.hidden_units))
                ),
                shape=[self.hidden_units, output_units]
            )
            b_out = tf.get_variable(
                name='b_out',
                initializer=tf.constant_initializer(0.0),
                shape=[output_units]
            )

            activation_function = self._get_activation_function()
            a_in = activation_function(tf.matmul(self.x, W_in) + b_in)
            z_out = tf.matmul(a_in, W_out) + b_out

            pis, sigmas, self.mus = tf.split(1, 3, z_out)
            self.pis = tf.nn.softmax(pis - tf.reduce_min(pis))
            self.sigmas = tf.exp(sigmas)

            self.loss = self._NLL(self.y, self.pis, self.mus, self.sigmas)
            optimizer = self._get_optimizer()(self.learning_rate)
            self.step = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()

        return graph

    def _get_activation_function(self):
        if self.activation_function == 'tanh':
            return tf.nn.tanh
        elif self.activation_function == 'sigmoid':
            return tf.nn.sigmoid
        elif self.activation_function == 'relu':
            return tf.nn.relu
        else:
            assert False, 'activation function must be tanh, sigmoid, or relu'

    def _get_optimizer(self):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer
        elif self.optimizer == 'ada':
            return tf.train.AdadeltaOptimizer
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer
        else:
            assert False, 'optimizer must be adam, ada, or gd'

    def _NLL(self, y, pis, mus, sigmas):
        norm = 1.0 / (sigmas * tf.sqrt(2 * np.pi))
        exp = -0.5 * tf.square((y - mus) / sigmas)
        gaussian_likelihoods = tf.exp(exp) * norm
        mixture_likelihoods = tf.reduce_sum(
            pis * gaussian_likelihoods, 1, keep_dims=True)
        return -tf.reduce_mean(tf.log(mixture_likelihoods + 1e-10))

    def _sample_mixture(self, pis, sigmas, mus):
        dist = pis / pis.sum()
        mixture_ind = np.random.choice(len(dist), size=1, p=dist)[0]
        mu = mus[mixture_ind]
        sigma = sigmas[mixture_ind]
        sample = np.random.normal(loc=mu, scale=sigma)
        return sample

    def _batch_generator(self, X, Y):
        while True:
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X, Y = X[idx], Y[idx]
            for i in range(1, len(X) - self.batch_size, self.batch_size):
                indices = idx[i: i + self.batch_size]
                yield X[indices], Y[indices]
