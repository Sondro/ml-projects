from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class DCGAN(object):

    def __init__(
        self,
        noise_shape,
        image_shape,
        num_training_steps=50000,
        warm_start_init_step=0,
        batch_size=128,
        k=2,
        verbose=False,
        save_images=False,
    ):
        self.noise_shape = noise_shape
        self.image_shape = image_shape
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        self.save_images = save_images

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.warm_start_init_step = warm_start_init_step
        self.build_graph()

    def build_graph(self):
        with self.graph.as_default():

            self.batch_norms = lambda_defaultdict(lambda x: batch_norm(name=x))

            self.x_noise = tf.placeholder(tf.float32, shape=self.noise_shape)
            self.x_image = tf.placeholder(tf.float32, shape=self.image_shape)
            self.x_synth = self.generator(self.x_noise)

            synth_preds = self.discriminator(self.x_synth)
            image_preds = self.discriminator(self.x_image, reuse=True)

            synth_loss = tf.log(tf.reduce_mean(1 - synth_preds))
            image_loss = tf.log(tf.reduce_mean(image_preds))

            vars_ = tf.trainable_variables()
            d_vars = [v for v in vars_ if v.name.startswith('discriminator/')]
            g_vars = [v for v in vars_ if v.name.startswith('generator/')]

            self.d_loss = -1.0 * (synth_loss + image_loss)
            self.g_loss = synth_loss

            optimizer = tf.train.AdamOptimizer(learning_rate=.0002, beta1=.5)
            self.d_step = optimizer.minimize(self.d_loss, var_list=d_vars)
            self.g_step = optimizer.minimize(self.g_loss, var_list=g_vars)

            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

    def generator(self, noise_samples, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):

            input_units = noise_samples.get_shape().as_list()[1]
            layers = [(4, 4, 32), (7, 7, 16), (14, 14, 8), (28, 28, 1)]

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
            z_fc = tf.matmul(noise_samples, W_fc) + b_fc
            z_fc = tf.reshape(z_fc,
                              [-1, layers[0][0], layers[0][1], layers[0][2]])
            a_fc = tf.nn.relu(self.batch_norms['g0'](z_fc))

            inputs = a_fc
            for i, layer in enumerate(layers[1:]):
                outputs = self.deconv2d(
                    inputs=inputs,
                    output_shape=[
                        self.batch_size, layer[0], layer[1], layer[2]
                    ],
                    name='deconv' + str(i)
                )
                if i == len(layers[1:]) - 1:
                    inputs = tf.nn.tanh(outputs)
                else:
                    norm = self.batch_norms['g{}'.format(i + 1)]
                    inputs = tf.nn.relu(norm(outputs))

            return inputs

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):

            a_conv1 = self.conv_layer(
                inputs=images,
                k_size=[8, 8, 1, 16],
                strides=[1, 2, 2, 1],
                name='conv1',
                normalizer=self.batch_norms['d0']
            )
            a_conv2 = self.conv_layer(
                inputs=a_conv1,
                k_size=[6, 6, 16, 32],
                strides=[1, 2, 2, 1],
                name='conv2',
                normalizer=self.batch_norms['d1']
            )

            flattened_units = 4*4*32
            a_conv2_flat = tf.reshape(a_conv2, [-1, flattened_units])
            W = tf.get_variable(
                name='W_fc',
                shape=[flattened_units, 1],
                initializer=tf.random_normal_initializer(
                    stddev=(1 / np.sqrt(flattened_units))
                    )
            )
            b = tf.get_variable(
                name='b_fc',
                shape=[1],
                initializer=tf.constant_initializer(0.0)
            )
            z_out = tf.matmul(a_conv2_flat, W) + b
            return tf.nn.sigmoid(z_out)

    def conv_layer(self, inputs, k_size, strides, name, normalizer=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                name='weights',
                shape=k_size,
                initializer=tf.truncated_normal_initializer(
                    stddev=0.02
                )
            )
            b = tf.get_variable(
                name='biases',
                shape=k_size[-1],
                initializer=tf.constant_initializer(value=0.0)
            )
            z = tf.nn.conv2d(inputs, W, strides=strides, padding='SAME') + b
            return self.lrelu(normalizer(z) if batch_norm else z)

    def deconv2d(self, inputs, output_shape, name, k_size=5, stride=2):
        with tf.variable_scope(name):
            k_shape = [
                k_size, k_size, output_shape[-1], inputs.get_shape()[-1]
            ]
            W = tf.get_variable(
                name='weights',
                shape=k_shape,
                initializer=tf.random_normal_initializer(stddev=0.02)
            )
            deconv = tf.nn.conv2d_transpose(
                value=inputs,
                filter=W,
                output_shape=output_shape,
                strides=[1, stride, stride, 1]
            )
            b = tf.get_variable(
                name='biases',
                shape=[output_shape[-1]],
                initializer=tf.constant_initializer(0.0)
            )
            return tf.reshape(deconv + b, deconv.get_shape())

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak*x)

    def noise_samples(self, num_samples):
        return np.random.uniform(
            low=-1.0,
            high=1.0,
            size=[num_samples, self.noise_shape[1]]
        )

    def train(self, X):
        self.session.run(self.init)

        if self.warm_start_init_step:
            self.restore()
        batch_gen = self.generate_batches(X)

        average_d_loss, average_g_loss = 0, 0
        for step_num in range(self.warm_start_init_step,
                              self.num_training_steps + 1):

            # update discriminator for k steps
            for _ in range(self.k):
                x = batch_gen.next()
                noise = self.noise_samples(self.batch_size)
                _, d_loss = self.session.run(
                    fetches=[self.d_step, self.d_loss],
                    feed_dict={self.x_noise: noise, self.x_image: x}
                )

            # update generator for a single step
            x = batch_gen.next()
            noise = self.noise_samples(self.batch_size)
            _, g_loss = self.session.run(
                fetches=[self.g_step, self.g_loss],
                feed_dict={self.x_noise: noise, self.x_image: x}
            )

            average_d_loss += d_loss
            average_g_loss += g_loss

            if step_num % 10 == 0 and self.verbose:
                print '[[step {:>7}]]\tD loss: {:.6f}\tG loss: {:.6f}'.format(
                    step_num,
                    average_d_loss / 10.0,
                    average_g_loss / 10.0
                )
                average_d_loss, average_g_loss = 0, 0

            if step_num % 1000 == 0 and self.save_images:
                noise = self.noise_samples(1)
                self.generate(noise, 'figure/sample_{}'.format(step_num))

            if step_num % 5000 == 0:
                self.save(step_num)

    def save_image(self, image, savefile):
        plt.axis('off')
        plt.imshow(image, vmin=-1, vmax=1, cmap='gray')
        plt.savefig(savefile)
        plt.cla()

    def generate(self, noise, image_name=None):
        num_images = noise.shape[0]
        num_batches = (num_images / self.batch_size) + 1
        padding_size = self.batch_size*num_batches - num_images
        padding = self.noise_samples(padding_size)
        padded_noise = np.concatenate([noise, padding], axis=0)

        all_samples = np.empty(
            shape=[0, self.image_shape[1], self.image_shape[2]])
        chunks = np.split(padded_noise, num_batches, axis=0)
        for sample_chunk in chunks:
            samples = self.session.run(
                fetches=self.x_synth,
                feed_dict={self.x_noise: sample_chunk}
            )
            samples = samples.squeeze()
            all_samples = np.concatenate([all_samples, samples], axis=0)
        all_samples = all_samples[:num_images]

        if image_name:
            for i, sample in enumerate(all_samples):
                self.save_image(sample, '{}-{}'.format(image_name, i))

        return all_samples

    def generate_batches(self, X):
        image_shape = self.image_shape
        image_shape = [dim or -1 for dim in image_shape]
        while True:
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            for i in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch_idx = idx[i: i + self.batch_size]
                x = X[batch_idx]
                yield x.reshape(image_shape)

    def save(self, step_num):
        self.saver.save(
            self.session,
            'checkpoints/model',
            global_step=step_num
        )

    def restore(self):
        self.saver.restore(
            self.session,
            'checkpoints/model-{}'.format(self.warm_start_init_step)
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
