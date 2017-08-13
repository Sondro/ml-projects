from collections import deque
import logging
import os
import pprint as pp

import tensorflow as tf
import numpy as np

from data_wrapper import DataReader


def dense(inputs, output_units, bias=True, activation=None, dropout=None, scope='dense', reuse=False):
    """
    Accepts a rank 2 or rank 3 tensor and appies a dense layer along the last dimension

    Args:
        inputs: 2D/3D Tensor to be multiplied by 2D weight matrix
        output_units: Number of output units
        activation: activation function
        dropout: dropout keep prob

    Returns:
        Outputs of dense layer

    """
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.get_variable(
            name='weights',
            initializer=normal_initializer(shape(inputs, -1)),
            shape=[shape(inputs, -1), output_units]
        )
        if rank(inputs) == 2:
            z = tf.matmul(inputs, W)
        elif rank(inputs) == 3:
            z = tf.einsum('ijk,kl->ijl', inputs, W)
        else:
            assert False
        if bias:
            b = tf.get_variable(
                name='biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = z + b
        z = activation(z) if activation else z
        z = tf.nn.dropout(z, dropout) if dropout else z
        return z


def normal_initializer(m):
    """
    Used to initialize a matrix of shape [m, n] with entries sampled from
    a normal with mean 0 and variance 1/sqrt(m).

    Args:
        m: to calculate variance 1/sqrt(m)

    Returns:
        A tf.initializer

    """
    return tf.random_normal_initializer(stddev=1.0 / np.sqrt(m))


def rank(tensor):
    """Get tensor rank as list"""
    return len(tensor.shape.as_list())


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if not dim:
        return tensor.shape.as_list()
    if dim:
        return tensor.shape.as_list()[dim]


def bidirectional_lstm(inputs, lengths, state_size, keep_prob, scope='bi-lstm', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        cell_fw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                initializer=normal_initializer(state_size),
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        cell_bw = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
            tf.contrib.rnn.core_rnn_cell.LSTMCell(
                state_size,
                initializer=normal_initializer(state_size),
                reuse=reuse
            ),
            output_keep_prob=keep_prob
        )
        outputs, (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
            inputs=inputs,
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            sequence_length=lengths,
            dtype=tf.float32
        )
        outputs = tf.concat(outputs, 2)
        output_state = tf.concat([output_fw.h, output_bw.h], axis=1)
        return outputs, output_state



class rMDN(object):
    """recurrent mixture density network for modeling handwriting strokes"""

    def __init__(
        self,
        num_mixture_components=20,
        num_training_steps=10**5,
        optimizer='rms',
        learning_rate=.0001,
        hidden_units=400,
        keep_prob=0.85,
        batch_size=64,
        t_steps=250,
        checkpoint_dir='checkpoints/',
        log_file='logs.txt',
        save_model_interval=2500,
        log_loss_interval=10,
        warm_start_init_step=0,
        loss_smoothing_window_size=100,
        is_train=True,
        verbose=True
    ):
        self.max_ascii_len = 25
        self.max_stroke_len = t_steps if is_train else 1
        self.t_steps = self.max_stroke_len
        self.num_mixture_components = num_mixture_components
        self.num_training_steps = num_training_steps
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.batch_size = batch_size if is_train else 1
        # self.t_steps = t_steps if is_train else 1
        self.keep_prob_scalar = keep_prob
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        self.save_model_interval = save_model_interval
        self.log_loss_interval = log_loss_interval
        self.warm_start_init_step = warm_start_init_step
        self.loss_smoothing_window_size = loss_smoothing_window_size
        self.verbose = verbose

        self.cs=None

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())
        if verbose:
            logging.info('\nnew run with model:\n{}'.format(self))

        self._fit() if is_train else self._load_model()

    def _fit(self):
        print 'building graph'
        self.graph = self._build_graph()
        self.session = tf.Session(graph=self.graph)
        print 'built graph'

        with self.session.as_default():

            # restore saved model or initialize model parameters
            if self.warm_start_init_step:
                self._restore()
            else:
                self.session.run(self.init)

            reader = DataReader('data/lineStrokes/', self.max_stroke_len, self.max_ascii_len)

            train_generator = reader.train_batch_generator(self.batch_size)
            val_generator = reader.test_batch_generator(self.batch_size)

            loss_history = deque(maxlen=self.loss_smoothing_window_size)
            val_loss_history = deque(maxlen=self.loss_smoothing_window_size)
            for step in range(self.warm_start_init_step + 1,
                              self.num_training_steps + 1):

                # sample batch, evaluate loss, update parameters
                sample_x, sample_y, lengths, sample_c, c_lens = train_generator.next()

                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict={
                        self.x: sample_x,
                        self.y: sample_y,
                        self.lengths: lengths - 2,
                        self.c: sample_c,
                        self.c_len: c_lens,
                        self.keep_prob: self.keep_prob_scalar
                    }
                )
                loss_history.append(train_loss)

                sample_x, sample_y, lengths, sample_c, c_lens = val_generator.next()
                [val_loss] = self.session.run(
                    fetches=[self.loss],
                    feed_dict={
                        self.x: sample_x,
                        self.y: sample_y,
                        self.lengths: lengths - 2,
                        self.c: sample_c,
                        self.c_len: c_lens,
                        self.keep_prob: 1.0
                    }
                )
                val_loss_history.append(val_loss)

                # write loss to logs every log_loss_interval
                if step % self.log_loss_interval == 0:
                    avg_train_loss = sum(loss_history) / len(loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    log = '[[step {:>7}]]\ttrain loss: {:>7}, \tval loss: {:>7}'.format(step, avg_train_loss, avg_val_loss)
                    if self.verbose:
                        logging.info(log)

                # save model checkpoint every save_model_interval
                if step % self.save_model_interval == 0:
                    self._save(step)

    def _load_model(self):
        self.graph = self._build_graph()
        self.session = tf.Session(graph=self.graph)
        self._restore()

    def _init(self, units):
        return tf.random_normal_initializer(stddev=1.0/np.sqrt(units))

    def _build_graph(self):
        input_units = 3  # [x offset, y offset, end of stroke probability]
        output_units = self.num_mixture_components * \
            (        # for each gaussian:
                2 +     # 2 variance parameters (diagonal of covariance matrix)
                1 +     # 1 correlation parameter
                2 +     # 2 mean parameters
                1       # mixing coefficient
            ) + 1    # end of stroke probability

        with tf.Graph().as_default() as graph:

            self.lengths = tf.placeholder(
                dtype=tf.int32,
                shape=[None]
            )
            self.x = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None, input_units]
            )
            self.y = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None, 3]
            )
            self.c = tf.placeholder(
                dtype=tf.int32,
                shape=[None, 25]
            )
            self.c_len = tf.placeholder(
                dtype=tf.int32,
                shape=[None]
            )
            self.keep_prob = tf.placeholder(dtype=tf.float32)

            x = self.x


            cell_state_size = cell_output_size = self.hidden_units

            # lstm layer 1
            self.cell1 =  tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                tf.contrib.rnn.core_rnn_cell.LSTMCell(
                    self.hidden_units,
                    initializer=self._init(cell_state_size + input_units)
                ),
                output_keep_prob=self.keep_prob
            )
            self.s1_in = self.cell1.zero_state(self.batch_size, tf.float32)
            z1, self.s1_out = tf.nn.dynamic_rnn(
                inputs=x,
                initial_state=self.s1_in,
                cell=self.cell1,
                scope='lstm1',
                sequence_length=self.lengths
            )
            xz1 = tf.concat([x, z1], axis=2)





            c = tf.one_hot(self.c, 128)
            c_ = dense(c, 20, scope='char')
            # c__ = convolutional_embedding(c, 20, 3)
            c, _ = bidirectional_lstm(c_, self.c_len, 20, self.keep_prob, scope='bi-lstm')
            # c = tf.concat([c_, c__, c___], axis=2)

            self.K = 8
            abk = dense(xz1, 3*self.K)
            alpha, beta, kappa = tf.split(tf.expand_dims(tf.exp(abk)/25, 3), 3, axis=2)
            self.init_kappa = tf.zeros_like(kappa)
            self.delta = kappa
            self.kappa = self.init_kappa + tf.cumsum(kappa, axis=1)
            tf.summary.histogram('kappa', self.delta)


            enum = tf.reshape(tf.range(self.max_ascii_len), (1, 1, 1, self.max_ascii_len))
            u = tf.cast(tf.tile(enum, (self.batch_size, self.max_stroke_len, self.K, 1)), tf.float32)
            self.phi = tf.reduce_sum(alpha*tf.exp(-beta*tf.square(self.kappa - u)), axis=2)
            w = tf.matmul(self.phi, c)


            # lstm layer 2
            self.cell2 =  tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                tf.contrib.rnn.core_rnn_cell.LSTMCell(
                    self.hidden_units,
                    use_peepholes=True,
                    initializer=self._init(
                        cell_output_size + cell_state_size + input_units)
                ),
                output_keep_prob=self.keep_prob
            )
            self.s2_in = self.cell2.zero_state(self.batch_size, tf.float32)
            z2, self.s2_out = tf.nn.dynamic_rnn(
                inputs=tf.concat([x, z1, w], axis=2),
                initial_state=self.s2_in,
                cell=self.cell2,
                scope='lstm2',
                sequence_length=self.lengths
            )
            xz2 = tf.concat([x, z2], axis=2)

            print '3'

            # lstm layer 3
            self.cell3 =  tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                tf.contrib.rnn.core_rnn_cell.LSTMCell(
                    self.hidden_units,
                    use_peepholes=True,
                    initializer=self._init(
                        cell_output_size + cell_state_size + input_units)
                ),
                output_keep_prob=self.keep_prob
            )
            self.s3_in = self.cell3.zero_state(self.batch_size, tf.float32)
            z3, self.s3_out = tf.nn.dynamic_rnn(
                inputs=tf.concat([x, z2, w], axis=2),
                initial_state=self.s3_in,
                cell=self.cell3,
                scope='lstm3',
                sequence_length=self.lengths
            )

            z1z2z3 = tf.concat([z1, z2, z3], axis=2)
            a = z1z2z3


            print '4'
            # output layer
            W = tf.get_variable(
                name='fc_weights',
                initializer=self._init(3*self.cell3.output_size),
                shape=[3*self.cell3.output_size, output_units]
            )
            b = tf.get_variable(
                name='fc_biases',
                initializer=tf.constant_initializer(),
                shape=[output_units]
            )
            z = tf.matmul(tf.reshape(a, [-1, 3*self.cell3.output_size]), W) + b
            z = tf.reshape(z, [-1, tf.shape(a)[1], output_units])

            # partition output into mixture model parameters and calculate loss
            self.pis, self.sigmas, self.rhos, self.mus, self.es = \
                self._parse_parameters(z)

            self.loss = self._NLL(
                self.y, self.pis, self.mus, self.sigmas,
                self.rhos, self.es, self.lengths
            )
            for var in tf.global_variables():
                print var.name, var
            # calculate gradients and update model parameters
            optimizer = self._get_optimizer()
            grads = optimizer.compute_gradients(self.loss)
            clipped = [(tf.clip_by_value(g, -10, 10), v) for g, v in grads]
            self.step = optimizer.apply_gradients(clipped)

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

            return graph

    def _concat_outputs(self, output_states_list):
        outputs = []
        for output_states in zip(*output_states_list):
            concat = tf.concat(list(output_states), 1)
            outputs.append(concat)
        return outputs

    def _parse_parameters(self, z):
        unit = self.num_mixture_components

        # mixture coefficients: softmax ensures valid distribution
        pis = tf.slice(z, [0, 0, 0], [-1, -1, unit])
        self.raw_pi = pis
        pis = tf.nn.softmax(pis - tf.reduce_min(pis))

        # covariance parameters: sigmas - (0, 10), rho - (0, 1)
        # clipped for numerical stability and to ensure positive
        # semi-definite covariance matrix
        sigmas = tf.slice(z, [0, 0, unit], [-1, -1, 2*unit])
        sigmas = tf.clip_by_value(tf.exp(sigmas), 1e-5, 10)
        rhos = tf.slice(z, [0, 0, 3*unit], [-1, -1, unit])
        rhos = tf.clip_by_value(tf.tanh(rhos), -1.0 + 1e-5, 1.0 - 1e-5)

        # mean parameters: (-inf, inf)
        mus = tf.slice(z, [0, 0, 4*unit], [-1, -1,  2*unit])

        # end of stroke bernoulli parameter: (0, 1)
        es = tf.slice(z, [0, 0, 6*unit], [-1, -1,  1])
        es = 1.0 / (1 + tf.exp(es))

        return pis, sigmas, rhos, mus, es

    def _NLL(self, y, pis, mus, sigmas, rho, es, lengths):
        sigma_1, sigma_2 = tf.split(sigmas, 2, axis=2)
        y_1, y_2, y_3 = tf.split(y, 3, axis=2)
        mu_1, mu_2 = tf.split(mus, 2, axis=2)

        norm = 1.0 / (2*np.pi*sigma_1*sigma_2 * tf.sqrt(1 - tf.square(rho)))
        Z = tf.square((y_1 - mu_1) / (sigma_1)) + \
            tf.square((y_2 - mu_2) / (sigma_2)) - \
            2*rho*(y_1 - mu_1)*(y_2 - mu_2) / (sigma_1*sigma_2)

        exp = -1.0*Z / (2*(1 - tf.square(rho)))
        gaussian_likelihoods = tf.exp(exp) * norm
        likelihoods = tf.reduce_sum(pis * gaussian_likelihoods, 2)
        likelihoods = tf.clip_by_value(likelihoods, 1e-10, np.inf)

        eos_mask = tf.equal(tf.ones_like(y_3), y_3)
        bernoulli = tf.squeeze(tf.where(eos_mask, es, 1 - es))
        bernoulli = tf.clip_by_value(bernoulli, 1e-10, 1.0)

        nll = -(tf.log(likelihoods) + tf.log(bernoulli))
        sequence_mask = tf.sequence_mask(
            lengths, dtype=tf.float32, maxlen=self.t_steps)
        nll = nll * sequence_mask

        return tf.reduce_sum(nll) / tf.cast(tf.reduce_sum(lengths), tf.float32)

    def generate_sequence(self, c, c_lens, num_steps, bias=0.0, prime_strokes=None, prime_ascii=None, prime_ascii_lens=None):
        c_ins = [self.s1_in.c, self.s2_in.c, self.s3_in.c]
        c_outs = [self.s1_out.c, self.s2_out.c, self.s3_out.c]
        h_ins = [self.s1_in.h, self.s2_in.h, self.s3_in.h]
        h_outs = [self.s1_out.h, self.s2_out.h, self.s3_out.h]
        params = [self.raw_pi, self.pis, self.mus, self.sigmas, self.rhos, self.es]

        c1, c2, c3 = self.session.run(c_ins)
        h1, h2, h3 = self.session.run(h_ins)
        kappa = np.zeros([self.batch_size, self.max_stroke_len, 8, 1])

        if prime_strokes is not None:

            if not self.cs:
                for x_t in prime_strokes:
                    c1, c2, c3, h1, h2, h3, kappa = self.session.run(
                        fetches=c_outs + h_outs + [self.kappa],
                        feed_dict={
                            self.s1_in.c: c1,
                            self.s2_in.c: c2,
                            self.s3_in.c: c3,
                            self.s1_in.h: h1,
                            self.s2_in.h: h2,
                            self.s3_in.h: h3,
                            self.c: prime_ascii,
                            self.c_len: prime_ascii_lens,
                            self.x: [[x_t]],
                            self.keep_prob: 1.0,
                            self.lengths: [1],
                            self.init_kappa: kappa
                        }
                    )
                self.cs = [c1, c2, c3]
            else:
                c1, c2, c3 = self.cs

        x = [[0.0, 0.0, 1.0]]
        kappa = np.zeros([self.batch_size, self.max_stroke_len, 8, 1])
        h1, h2, h3 = self.session.run(h_ins)
        while True:
            c1, c2, c3, h1, h2, h3, raw_pi, pi, mu, sigma, rho, e, kappa, phi = self.session.run(
                fetches=c_outs + h_outs + params + [self.kappa, self.phi],
                feed_dict={
                    self.s1_in.c: c1,
                    self.s2_in.c: c2,
                    self.s3_in.c: c3,
                    self.s1_in.h: h1,
                    self.s2_in.h: h2,
                    self.s3_in.h: h3,
                    self.c: c,
                    self.c_len: c_lens,
                    self.x: [[x[-1]]],
                    self.keep_prob: 1.0,
                    self.lengths: [1],
                    self.init_kappa: kappa
                }
            )
            x_t = self._sample_mixture(raw_pi, pi, sigma, mu, rho, e, bias).squeeze()
            x.append(x_t)
            print phi.squeeze().argmax(), c_lens[0]
            if phi.squeeze().argmax() > c_lens[0]:
                break

        return np.array(x)

    def _sample_mixture(self, raw_pis, pis, sigmas, mus, rhos, e, bias):
        pis = pis.flatten()


        raw_pis = raw_pis.flatten()
        raw_pis = np.exp(raw_pis + bias)
        # raw_pis = raw_pis - max(raw_pis)
        raw_pis = raw_pis / raw_pis.sum()



        sigmas = sigmas.flatten() * np.exp(-bias)
        sigmas_1, sigmas_2 = np.split(sigmas, 2)
        mus = mus.flatten()
        mus_1, mus_2 = np.split(mus, 2)
        rhos = rhos.flatten()
        e = e.flatten()[0]

        dist = raw_pis / raw_pis.sum()
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
        sample_e = 1 if np.random.rand() < e else 0
        return np.array([[sample[0], sample[1], sample_e]])

    def _get_optimizer(self):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate)
        elif self.optimizer == 'ada':
            return tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(
                self.learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, ada, gd, or rms'

    def _save(self, step_num):
        if not os.path.isdir(self.checkpoint_dir):
            if self.verbose:
                logging.info(
                    'creating checkpoint directory {}'.format(
                        self.checkpoint_dir
                    )
                )
                os.mkdir(self.checkpoint_dir)

        model_path = os.path.join(self.checkpoint_dir, 'model')
        if self.verbose:
            logging.info('saving model to {}'.format(model_path))
        self.saver.save(self.session, model_path, global_step=step_num)

    def _restore(self):
        if not self.warm_start_init_step:
            model_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                self.checkpoint_dir,
                'model-{}'.format(self.warm_start_init_step)
            )
            if self.verbose:
                logging.info('restoring model from {}'.format(model_path))
            self.saver.restore(self.session, model_path)

    def __str__(self):
        return pp.pformat(self.__dict__)

if __name__ == '__main__':
    model = rMDN(num_training_steps=100000) # loss @100k: 4.35
