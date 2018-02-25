import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from utils import *



class DCGAN():

    def __init__(self, sess, batch_size = 100, lr_rate = 0.0002, epochs = 100, keep_prob = 0.3, random_dim = 100):

        # training hyperparameters
        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.keep_prob = keep_prob
        self.random_dim = random_dim

        # generate fixed inputs form gaussian normal
        self.fixed_z_ = np.random.normal(0, 1, (25, 1, 1, random_dim))

        # tensorflow session object 
        self.sess = sess

        # construct model
        self.__construct_model()
        sess.run(tf.global_variables_initializer())


    def __generator(self, x, isTrain=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):

            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(x, 128, [4, 4], strides=(1, 1), padding='valid')
            lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

            # output layer
            conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv5)

            return o
    
    def __discriminator(self, x, isTrain=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # 1st hidden layer
            conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu1 = lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 64, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

            # output layer
            conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
            #o = tf.nn.sigmoid(conv5)

            return conv5
    
       
    def __construct_model(self):

        # variables : input
        self._z = tf.placeholder(tf.float32, shape=(None, 1, 1, self.random_dim))
        self._x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
        self._isTrain = tf.placeholder(dtype=tf.bool)

        # networks : generator
        self._G_z = self.__generator(self._z, self._isTrain)

        # networks : discriminator
        D_real_logits = self.__discriminator(self._x, self._isTrain)
        D_fake_logits = self.__discriminator(self._G_z, self._isTrain, reuse=True)


        # loss for each network
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([self.batch_size, 1, 1, 1])))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([self.batch_size, 1, 1, 1])))
        self._D_loss = D_loss_real + D_loss_fake
        self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([self.batch_size, 1, 1, 1])))

        # trainable variables for each network
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        G_vars = [var for var in T_vars if var.name.startswith('generator')]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self._D_optim = tf.train.AdamOptimizer(self.lr_rate, beta1=0.5).minimize(self._D_loss, var_list=D_vars)
            self._G_optim = tf.train.AdamOptimizer(self.lr_rate, beta1=0.5).minimize(self._G_loss, var_list=G_vars)


    def fit(self, train_set):

        train_var = [self._z, self._G_z]
        tf.add_to_collection('train_var', train_var[0])
        tf.add_to_collection('train_var', train_var[1])
     
        saver = tf.train.Saver(max_to_keep=25)
        saver.export_meta_graph('models/dcgan_model_ckpt.meta', collection_list=['train_var'])
       

        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []
        

        # training-loop
        print('training start!')
        start_time = time.time()
        for epoch in range(self.epochs):
            G_losses = []
            D_losses = []
            epoch_start_time = time.time()
            idx = np.arange(train_set.shape[0])
            np.random.shuffle(idx)
            for iter in tqdm(range(train_set.shape[0] // self.batch_size)):
                # update discriminator
                x_ = train_set[idx[iter*self.batch_size:(iter+1)*self.batch_size]]
                z_ = np.random.normal(0, 1, (x_.shape[0], 1, 1, self.random_dim))
                loss_d_, _ = self.sess.run([self._D_loss, self._D_optim], {self._x: x_, self._z: z_, self._isTrain: True})
                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (x_.shape[0], 1, 1, self.random_dim))
                loss_g_, _ = self.sess.run([self._G_loss, self._G_optim], {self._z: z_, self._x: x_, self._isTrain: True})
                G_losses.append(loss_g_)

            if (epoch+1) % 10 == 0:
                saver.save(self.sess, 'models/dcgan_model_ckpt', global_step=epoch+1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), self.epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            p = 'results/dcgan_random_results/generated_image_epoch_' + str(epoch + 1) + '.png'
            fixed_p = 'results/dcgan_fixed_results/generated_image_epoch_' + str(epoch + 1) + '.png'
            save_results(self, (epoch + 1), path=p, isFix=False, img_size=(64, 64))
            save_results(self, (epoch + 1), path=fixed_p, isFix=True, img_size=(64, 64))
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), self.epochs, total_ptime))
        print("Training finish!... save training results")
        with open('results/dcgan_train_history.pkl', 'wb') as f:
            pickle.dump(train_hist, f)
        save_train_history(train_hist,  path='results/dcgan_train_history.png')
        save_animation(self.epochs, 'results/dcgan_animation.gif', 'results/dcgan_fixed_results/')

    def generate(self, isFix = False, number = 1):
        if isFix:
            return self.sess.run(self._G_z, {self._z: self.fixed_z_, self._isTrain: False})
        z = np.random.normal(0, 1, (number, 1, 1, self.random_dim))
        return self.sess.run(self._G_z, {self._z: z, self._isTrain: False})


    def load(self, epoch):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'models/dcgan_model_ckpt-' + str(epoch))