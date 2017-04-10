import numpy as np
import tensorflow as tf


class MDN(object):

    def __init__(
        self,
        num_mixture_components=8,
        num_training_steps=10**5,
        activation_function='tanh',  # 'tanh', 'sigmoid', or 'relu'
        optimizer='adam',  # 'adam', 'gd', or 'ada'
        learning_rate=.01,
        hidden_units=32,
        batch_size=128,
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
        output_units = self.num_mixture_components * \
            (        # for each gaussian...
                2 +  # 2 variance parameters (diagonal of covariance matrix)
                1 +  # 1 correlation parameter
                2 +  # 2 mean parameters
                1    # mixing coefficient
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
        pis, sigmas, mus, rhos = self.session.run(
            [self.pis, self.sigmas, self.mus, self.rhos],
            feed_dict={self.x: X}
        )
        Y = []
        for pi, sigma, mu, rho in zip(pis, sigmas, mus, rhos):
            sample = self._sample_mixture(pi, sigma, mu, rho)
            Y.append(sample)
        return np.array(Y).squeeze()

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
                shape=[input_units, self.hidden_units]
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

            self.pis, self.sigmas, self.rhos, self.mus = \
                self._parse_parameters(z_out)

            self.loss = self._NLL(
                self.y, self.pis, self.mus, self.sigmas, self.rhos)
            optimizer = self._get_optimizer()(self.learning_rate)
            self.step = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()

        return graph

    def _parse_parameters(self, z_out):
        unit = self.num_mixture_components

        pis = tf.slice(z_out, [0, 0], [-1, unit])
        pis = tf.nn.softmax(pis - tf.reduce_min(pis))

        sigmas = tf.slice(z_out, [0, unit], [-1, 2*unit])
        sigmas = tf.exp(sigmas)

        rhos = tf.slice(z_out, [0, 3*unit], [-1, unit])
        rhos = tf.tanh(rhos)

        mus = tf.slice(z_out, [0, 4*unit], [-1, 2*unit])
        return pis, sigmas, rhos, mus

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

    def __NLL(self, y, pis, mus, sigmas):
        norm = 1.0 / (sigmas * tf.sqrt(2 * np.pi))
        exp = -0.5 * tf.square((y - mus) / sigmas)
        gaussian_likelihoods = tf.exp(exp) * norm
        mixture_likelihoods = tf.reduce_sum(
            pis * gaussian_likelihoods, 1, keep_dims=True)
        return -tf.reduce_mean(tf.log(mixture_likelihoods + 1e-10))

    def _NLL(self, y, pis, mus, sigmas, rho):
        sigma_1, sigma_2 = tf.split(1, 2, sigmas)
        y_1, y_2 = tf.split(1, 2, y)
        mu_1, mu_2 = tf.split(1, 2, mus)

        norm = 1.0 / (2*np.pi*sigma_1*sigma_2 * tf.sqrt(1 - tf.square(rho)))
        Z = tf.square((y_1 - mu_1) / sigma_1) + \
            tf.square((y_2 - mu_2) / sigma_2) - \
            2*rho*(y_1 - mu_1)*(y_2 - mu_2) / (sigma_1*sigma_2)
        exp = -1.0*Z / (2*(1 - tf.square(rho)))
        gaussian_likelihoods = tf.exp(exp) * norm
        mixture_likelihoods = tf.reduce_sum(pis * gaussian_likelihoods, 1)

        return -tf.reduce_mean(tf.log(mixture_likelihoods + 1e-10))

    def _sample_mixture(self, pis, sigmas, mus, rhos):
        pis = pis.flatten()
        sigmas = sigmas.flatten()
        sigmas_1, sigmas_2 = np.split(sigmas, 2)
        mus = mus.flatten()
        mus_1, mus_2 = np.split(mus, 2)
        rhos = rhos.flatten()

        dist = pis / pis.sum()
        mixture_ind = np.random.choice(len(dist), size=1, p=dist)[0]

        mu_1 = mus_1[mixture_ind]
        mu_2 = mus_2[mixture_ind]
        sigma_1 = sigmas_1[mixture_ind]
        sigma_2 = sigmas_2[mixture_ind]
        rho = rhos[mixture_ind]

        covariance_matrix = np.array(
            [[sigma_1**2, rho*sigma_1*sigma_2],
             [rho*sigma_1*sigma_2, sigma_2**2]]
        )

        mean_vector = np.array([mu_1, mu_2])
        sample = np.random.multivariate_normal(mean_vector, covariance_matrix)
        return np.array([[sample[0], sample[1]]])

    def _batch_generator(self, X, Y):
        while True:
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X, Y = X[idx], Y[idx]
            for i in range(1, len(X) - self.batch_size, self.batch_size):
                indices = idx[i: i + self.batch_size]
                yield X[indices], Y[indices]
