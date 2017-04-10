import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os


class ACGAN(object):

    def __init__(
        self,
        noise_shape,
        image_shape,
        y_shape,
        num_training_steps=50000,
        warm_start_init_step=0,
        learning_rate=.0002,
        keep_prob=1.0,
        batch_size=128,
        k=1,
        verbose=False,
        save_images=False,
        checkpoint_dir=None
    ):
        self.noise_shape = noise_shape
        self.image_shape = image_shape
        self.y_shape = y_shape
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.keep_prob_scalar = keep_prob
        self.k = k
        self.verbose = verbose
        self.save_images = save_images
        self.checkpoint_dir = checkpoint_dir

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.warm_start_init_step = warm_start_init_step
        self.build_graph()

    def build_graph(self):

        with self.graph.as_default():

            self.batch_norms = lambda_defaultdict(lambda x: batch_norm(name=x))
            self.keep_prob = tf.placeholder(dtype=tf.float32)

            self.x_noise = tf.placeholder(tf.float32, shape=self.noise_shape)
            self.y_noise = tf.placeholder(tf.float32, shape=self.y_shape)
            self.x_image = tf.placeholder(tf.float32, shape=self.image_shape)
            self.y_image = tf.placeholder(tf.float32, shape=self.y_shape)

            self.x_synth = self.generator(self.x_noise, self.y_noise)

            snyth_prob, synth_logits = self.discriminator(self.x_synth)
            image_prob, image_logits = self.discriminator(self.x_image, reuse=True)

            synth_loss = tf.log(tf.reduce_mean(1 - snyth_prob))
            image_loss = tf.log(tf.reduce_mean(image_prob))

            self.synth_ce = self.cross_entropy(synth_logits, self.y_noise)
            self.image_ce = self.cross_entropy(image_logits, self.y_image)
            self.d_loss = -1.0 * (synth_loss + image_loss) + self.image_ce
            self.g_loss = synth_loss + self.synth_ce

            vars_ = tf.trainable_variables()
            d_vars = [v for v in vars_ if v.name.startswith('discriminator/')]
            g_vars = [v for v in vars_ if v.name.startswith('generator/')]

            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=.5)
            self.d_step = optimizer.minimize(self.d_loss, var_list=d_vars)
            self.g_step = optimizer.minimize(self.g_loss, var_list=g_vars)

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def cross_entropy(self, logits, targets):
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, targets))

    def generator(self, noise_samples, y, reuse=False):

        with tf.variable_scope('generator', reuse=reuse):

            inputs = tf.concat(1, [noise_samples, y])
            input_units = inputs.get_shape().as_list()[1]
            layers = [(4, 4, 128), (7, 7, 64), (14, 14, 32), (28, 28, 1)]

            y_dim = y.get_shape().as_list()[1]
            y_depth = tf.reshape(y, [-1, 1, 1, y_dim])

            W_fc = tf.get_variable(
                name='W_fc',
                shape=[input_units, layers[0][0]*layers[0][1]*layers[0][2]],
                initializer=tf.random_normal_initializer(
                    stddev=(1 / np.sqrt(input_units))
                )
            )
            b_fc = tf.get_variable(
                name='b_fc',
                shape=[layers[0][0]*layers[0][1]*layers[0][2]],
                initializer=tf.constant_initializer(0.0)
            )
            z_fc = tf.matmul(inputs, W_fc) + b_fc
            a_fc = tf.nn.relu(z_fc)

            inputs = tf.reshape(a_fc, [-1, layers[0][0], layers[0][1], layers[0][2]])
            for i, layer in enumerate(layers[1:]):
                is_last = i == len(layers[1:]) - 1

                # concatenate conditioning vector along depth axis
                input_shape = inputs.get_shape().as_list()
                y_concat = tf.tile(y_depth, [1, input_shape[1], input_shape[2], 1])
                inputs = tf.concat(3, [inputs, y_concat])

                output_shape = [self.batch_size, layer[0], layer[1], layer[2]]
                inputs = self.deconv_layer(
                    inputs=inputs,
                    output_shape=output_shape,
                    name='deconv{}'.format(i),
                    normalizer=None if is_last
                                    else self.batch_norms['g{}'.format(i)],
                    dropout=None,
                    activation=tf.nn.tanh if is_last else tf.nn.relu
                )

            return inputs

    def deconv_layer(self, inputs, output_shape, name, ksize=5,
                     stride=2, dropout=None, normalizer=None, activation=None):

        with tf.variable_scope(name):

            k_shape = [ksize, ksize, output_shape[-1], inputs.get_shape()[-1]]
            W = tf.get_variable(
                name='weights',
                shape=k_shape,
                initializer=tf.random_normal_initializer(stddev=0.02)
            )
            b = tf.get_variable(
                name='biases',
                shape=[output_shape[-1]],
                initializer=tf.constant_initializer(0.0)
            )
            deconv = tf.nn.conv2d_transpose(
                value=inputs,
                filter=W,
                output_shape=output_shape,
                strides=[1, stride, stride, 1]
            )
            z = tf.reshape(deconv + b, deconv.get_shape())

            if dropout is not None:
                z = tf.nn.dropout(z, keep_prob=dropout)
            if normalizer is not None:
                z = normalizer(z)
            if activation is not None:
                z = activation(z)

            return z

    def discriminator(self, images, reuse=False):

        with tf.variable_scope('discriminator', reuse=reuse):

            depths = [1, 16, 32, 64, 128]
            strides = [2, 2, 1, 2, 2]

            inputs = images
            depth_pairs = zip(depths[:-1], depths[1:])
            for i, (depth_pair, stride) in enumerate(zip(depth_pairs, strides)):
                is_first = i == 0
                inputs = self.conv_layer(
                    inputs=inputs,
                    ksize=[3, 3, depth_pair[0], depth_pair[1]],
                    strides=[1, stride, stride, 1],
                    name='conv{}'.format(i),
                    dropout=self.keep_prob,
                    normalizer=None if is_first
                                    else self.batch_norms['d{}'.format(i)],
                    activation=self.lrelu
                )

            flattened_units = 4*4*128
            conv_out = tf.reshape(inputs, [-1, flattened_units])
            W = tf.get_variable(
                name='W_fc',
                shape=[conv_out.get_shape().as_list()[1], self.y_shape[1] + 1],
                initializer=tf.random_normal_initializer(
                    stddev=(1 / np.sqrt(flattened_units))
                    )
            )
            b = tf.get_variable(
                name='b_fc',
                shape=[self.y_shape[1] + 1],
                initializer=tf.constant_initializer(0.0)
            )
            z_out = tf.matmul(conv_out, W) + b
            prob_logit = tf.slice(z_out, [0, 0], [-1, 1])
            logits = tf.slice(z_out, [0, 1], [-1, self.y_shape[1]])
            return tf.nn.sigmoid(prob_logit), logits

    def conv_layer(self, inputs, ksize, strides, name,
                   dropout=None, normalizer=None, activation=None):

        with tf.variable_scope(name):

            W = tf.get_variable(
                name='weights',
                shape=ksize,
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            b = tf.get_variable(
                name='biases',
                shape=ksize[-1],
                initializer=tf.constant_initializer(value=0.0)
            )
            z = tf.nn.conv2d(inputs, W, strides=strides, padding='SAME') + b

            if dropout is not None:
                z = tf.nn.dropout(z, keep_prob=dropout)
            if normalizer is not None:
                z = normalizer(z)
            if activation is not None:
                z = activation(z)

            return z

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak*x)

    def noise_samples(self, num_samples):
        rand_int = np.random.randint(0, self.y_shape[1], size=num_samples)
        one_hot = np.zeros(shape=[num_samples, self.y_shape[1]])
        one_hot[np.arange(num_samples), rand_int] = 1
        noise = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=[num_samples, self.noise_shape[1]]
        )
        x_noise, y_noise = noise, one_hot
        return x_noise, y_noise

    def train(self, X, Y):
        self.session.run(self.init)
        if self.warm_start_init_step:
            self.restore()

        batch_gen = self.generate_batches(X, Y)
        average_d_loss, average_g_loss = 0, 0
        average_synth_ce, average_image_ce = 0, 0
        for step_num in range(self.warm_start_init_step,
                              self.num_training_steps + 1):

            for _ in range(self.k):
                x, y = batch_gen.next()
                x_noise, y_noise = self.noise_samples(self.batch_size)
                _, d_loss, image_ce = self.session.run(
                    fetches=[self.d_step, self.d_loss, self.image_ce],
                    feed_dict={
                        self.x_noise: x_noise,
                        self.x_image: x,
                        self.y_image: y,
                        self.y_noise: y_noise,
                        self.keep_prob: self.keep_prob_scalar
                    }
                )

            x_noise, y_noise = self.noise_samples(self.batch_size)
            _, g_loss, synth_ce = self.session.run(
                fetches=[self.g_step, self.g_loss, self.synth_ce],
                feed_dict={
                    self.x_noise: x_noise,
                    self.y_noise: y_noise,
                    self.keep_prob: self.keep_prob_scalar
                }
            )

            average_d_loss += d_loss
            average_g_loss += g_loss
            average_synth_ce += synth_ce
            average_image_ce += image_ce

            if step_num % 10 == 0 and self.verbose:
                with open('log.txt', 'a') as f:
                    metric_str_fmt = (
                        "[[step {:>7}]]\t"
                        "D loss: {:.6f}\t"
                        "G loss: {:.6f}\t"
                        "image_ce: {:.6f}\t"
                        "synth_ce: {:.6f}"
                    )
                    metric_str = metric_str_fmt.format(
                        step_num,
                        average_d_loss / 10.0,
                        average_g_loss / 10.0,
                        average_image_ce / 10.0,
                        average_synth_ce / 10.0
                    )
                    print metric_str
                    f.write(metric_str + '\n')

                average_d_loss, average_g_loss = 0, 0
                average_synth_ce, average_image_ce = 0, 0

            if step_num % 500 == 0 and self.save_images:
                x_noise, y_noise = self.noise_samples(1)
                image_dir = 'training_figures/'
                if not os.path.isdir(image_dir):
                    os.mkdir(image_dir)
                self.generate(x_noise, y_noise,
                              '{}sample_{}'.format(image_dir, step_num))

            if step_num % 5000 == 0:
                self.save(step_num)

    def save_image(self, image, savefile, title=None):
        plt.axis('off')
        image = image.squeeze()
        plt.imshow(image, vmin=-1, vmax=1, cmap='gray')
        if title:
            plt.title(title)
        plt.savefig(savefile)
        plt.cla()

    def generate(self, x_noise, y_noise, image_name=None):
        num_images = x_noise.shape[0]
        num_batches = (num_images / self.batch_size) + 1
        padding_size = self.batch_size*num_batches - num_images
        x_padding, y_padding = self.noise_samples(padding_size)
        padded_x_noise = np.concatenate([x_noise, x_padding], axis=0)
        padded_y_noise = np.concatenate([y_noise, y_padding], axis=0)

        all_samples = np.empty(
            shape=[0, self.image_shape[1], self.image_shape[2], self.image_shape[3]])
        x_chunks = np.split(padded_x_noise, num_batches, axis=0)
        y_chunks = np.split(padded_y_noise, num_batches, axis=0)
        for x_chunk, y_chunk in zip(x_chunks, y_chunks):
            samples = self.session.run(
                fetches=self.x_synth,
                feed_dict={
                    self.x_noise: x_chunk,
                    self.y_noise: y_chunk,
                    self.keep_prob: self.keep_prob_scalar
                }
            )
            all_samples = np.concatenate([all_samples, samples], axis=0)
        all_samples = all_samples[:num_images]

        if image_name:
            for (i, sample), y in zip(enumerate(all_samples), y_noise):
                self.save_image(sample, '{}-{}'.format(image_name, i),
                                title=str(y_noise.argmax()))

        return all_samples

    def generate_batches(self, X, Y):
        image_shape = self.image_shape
        image_shape = [dim or -1 for dim in image_shape]
        while True:
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for i in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch_idx = idx[i: i + self.batch_size]
                x = X[batch_idx]
                y = Y[batch_idx]
                yield x.reshape(image_shape), y

    def save(self, step_num):
        if self.checkpoint_dir:
            if not os.path.isdir(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
            self.saver.save(
                self.session,
                os.path.join(self.checkpoint_dir, 'model'),
                global_step=step_num
            )

    def restore(self):
        self.saver.restore(
            self.session,
            os.path.join(
                self.checkpoint_dir,
                'model-{}'.format(self.warm_start_init_step)
            )
        )


class lambda_defaultdict(defaultdict):

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            inputs=x,
            decay=self.momentum,
            updates_collections=None,
            epsilon=self.epsilon,
            scale=True,
            is_training=train,
            scope=self.name
        )
