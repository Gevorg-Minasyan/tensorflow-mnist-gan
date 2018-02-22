import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from utils import *



class GAN():

    def __init__(self, batch_size = 100, lr_rate = 0.0002, epochs = 100, keep_prob = 0.3, random_dim = 100):

        # training hyperparameters
        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.epochs = epochs
        self.keep_prob = keep_prob
        self.fixed_z_ = np.random.normal(0, 1, (25, random_dim))
        self.random_dim = random_dim


    def __generator(self, x):
            
        # initializers
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        w_init_xavier = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('G_w0', [x.get_shape()[1], 256], initializer=w_init)
        b0 = tf.get_variable('G_b0', [256], initializer=b_init)
        h0 = lrelu(tf.matmul(x, w0) + b0)

        # 2nd hidden layer
        w1 = tf.get_variable('G_w1', [h0.get_shape()[1], 512], initializer=w_init_xavier)
        b1 = tf.get_variable('G_b1', [512], initializer=b_init)
        h1 = lrelu(tf.matmul(h0, w1) + b1)

        # 3rd hidden layer
        w2 = tf.get_variable('G_w2', [h1.get_shape()[1], 1024], initializer=w_init_xavier)
        b2 = tf.get_variable('G_b2', [1024], initializer=b_init)
        h2 = lrelu(tf.matmul(h1, w2) + b2)

        # output hidden layer
        w3 = tf.get_variable('G_w3', [h2.get_shape()[1], 784], initializer=w_init_xavier)
        b3 = tf.get_variable('G_b3', [784], initializer=b_init)
        o = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        return o
    
    def __discriminator(self, x, drop_out):

        # initializers
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        w_init_xavier = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('D_w0', [x.get_shape()[1], 1024], initializer=w_init)
        b0 = tf.get_variable('D_b0', [1024], initializer=b_init)
        h0 = lrelu(tf.matmul(x, w0) + b0)
        h0 = tf.nn.dropout(h0, drop_out)

        # 2nd hidden layer
        w1 = tf.get_variable('D_w1', [h0.get_shape()[1], 512], initializer=w_init_xavier)
        b1 = tf.get_variable('D_b1', [512], initializer=b_init)
        h1 = lrelu(tf.matmul(h0, w1) + b1)
        h1 = tf.nn.dropout(h1, drop_out)

        # 3rd hidden layer
        w2 = tf.get_variable('D_w2', [h1.get_shape()[1], 256], initializer=w_init_xavier)
        b2 = tf.get_variable('D_b2', [256], initializer=b_init)
        h2 = lrelu(tf.matmul(h1, w2) + b2)
        h2 = tf.nn.dropout(h2, drop_out)

        # output layer
        w3 = tf.get_variable('D_w3', [h2.get_shape()[1], 1], initializer=w_init_xavier)
        b3 = tf.get_variable('D_b3', [1], initializer=b_init)
        o = tf.matmul(h2, w3) + b3

        return o
    
       


    def __construct_model(self):
        # networks : generator
        with tf.variable_scope('G'):
            self._z = tf.placeholder(tf.float32, shape=(None, self.random_dim))
            self._fake_y = tf.placeholder(tf.float32, shape=(None))
            self._G_z = self.__generator(self._z)

        # networks : discriminator
        with tf.variable_scope('D') as scope:
            self._drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
            self._real_y = tf.placeholder(tf.float32, shape=(None))
            self._x = tf.placeholder(tf.float32, shape=(None, 784))
            D_real = self.__discriminator(self._x, self._drop_out)
            scope.reuse_variables()
            D_fake = self.__discriminator(self._G_z, self._drop_out)


        # loss for each network
        #eps = 1e-2
        #self._D_loss = tf.reduce_mean(-tf.log(D_real+eps) - tf.log(1 - D_fake+eps))
        #self._G_loss = tf.reduce_mean(-tf.log(D_fake+eps))
        
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=self._real_y))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=self._fake_y))
        self._D_loss = D_loss_real + D_loss_fake
        self._G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=self._real_y))
        

        # trainable variables for each network
        t_vars = tf.trainable_variables()
        D_vars = [var for var in t_vars if 'D_' in var.name]
        G_vars = [var for var in t_vars if 'G_' in var.name]

        # optimizer for each network
        self._D_optim = tf.train.AdamOptimizer(self.lr_rate, beta1 = 0.5).minimize(self._D_loss, var_list=D_vars)
        self._G_optim = tf.train.AdamOptimizer(self.lr_rate, beta1 = 0.5).minimize(self._G_loss, var_list=G_vars)


    def fit(self, train_set):

        self.__construct_model()

        # open session and initialize all variables
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_var = [self._z, self._G_z, self._drop_out]
        tf.add_to_collection('train_var', train_var[0])
        tf.add_to_collection('train_var', train_var[1])
        tf.add_to_collection('train_var', train_var[2])
     
        saver = tf.train.Saver(max_to_keep=25)
        saver.export_meta_graph('models/gan_model_ckpt.meta', collection_list=['train_var'])
       

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
                #x_ = train_set[np.random.randint(0, train_set.shape[0], size=self.batch_size)]
                z_ = np.random.normal(0, 1, (x_.shape[0], self.random_dim))
                y_real_ = np.ones(x_.shape[0])*0.9
                y_fake_ = np.zeros(x_.shape[0])
                loss_d_, _ = self.sess.run([self._D_loss, self._D_optim], {self._x: x_, self._z: z_, self._real_y: y_real_, self._fake_y: y_fake_, self._drop_out: self.keep_prob})

                D_losses.append(loss_d_)

                # update generator
                z_ = np.random.normal(0, 1, (x_.shape[0], self.random_dim))
                y_real_ = np.ones(x_.shape[0])
                loss_g_, _ = self.sess.run([self._G_loss, self._G_optim], {self._z: z_, self._real_y: y_real_, self._drop_out: self.keep_prob})
                G_losses.append(loss_g_)

            if (epoch+1) % 20 == 0:
                saver.save(self.sess, 'models/gan_model_ckpt', global_step=epoch+1)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), self.epochs, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            p = 'results/random_results/gan_generated_image_epoch_' + str(epoch + 1) + '.png'
            fixed_p = 'results/fixed_results/gan_generated_image_epoch_' + str(epoch + 1) + '.png'
            save_results(self, self.random_dim, (epoch + 1), path=p, isFix=False)
            save_results(self, self.random_dim, (epoch + 1), path=fixed_p, isFix=True)
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), self.epochs, total_ptime))
        print("Training finish!... save training results")
        with open('results/train_history.pkl', 'wb') as f:
            pickle.dump(train_hist, f)
        save_train_history(train_hist,  path='results/train_history.png')
        save_animation(self.epochs)

    def generate(self, z = None):
        if z is None:
            return self.sess.run(self._G_z, {self._z: self.fixed_z_, self._drop_out: 0.0})
        return self.sess.run(self._G_z, {self._z: z, self._drop_out: 0.0})


    def load(self, epoch):
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.import_meta_graph('models/gan_model_ckpt.meta')
        saver.restore(self.sess, 'models/gan_model_ckpt-' + str(epoch))
        self._z = tf.get_collection('train_var')[0]
        self._G_z = tf.get_collection('train_var')[1]
        self._drop_out = tf.get_collection('train_var')[2]

