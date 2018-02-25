import argparse
from gan import GAN
from dcgan import DCGAN
import numpy as np
import tensorflow as tf
from utils import create_results_dir, save_results, save_animation


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--model', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--keep_prob', type=float, default=0.3)
    parser.add_argument('--restoring_epoch', type=int, default=200)
    parser.add_argument('--random_dim', type=int, default=100)
    args = parser.parse_args(*argument_array)
    return args

if __name__ == "__main__":

    args = parse_args()

    with tf.Session() as sess:

        if args.model == 0:
            model = GAN(sess = sess,
                        epochs = args.epochs, 
                        lr_rate = args.lr_rate,
                        batch_size = args.batch_size, 
                        keep_prob = args.keep_prob,
                        random_dim = args.random_dim)
        elif args.model == 1:
            model = DCGAN(sess = sess,
                          epochs = args.epochs, 
                          lr_rate = args.lr_rate,
                          batch_size = args.batch_size, 
                          keep_prob = args.keep_prob,
                          random_dim = args.random_dim)
        else:
            raise ValueError('The mode argument only takes values 0 or 1.')           

        if args.train == 'True':
            create_results_dir()

            # load MNIST
            X_train = np.load('MNIST_data/mnist_train_x.npy')
            
            if args.model == 0:  
                X_train = X_train.reshape(60000, 784)
            elif args.model == 1:
                X_train = X_train.reshape(60000, 28, 28, 1)
                X_train = tf.image.resize_images(X_train, [64, 64]).eval()

            # normalize -1 to 1
            X_train = (X_train.astype(np.float32) - 127.5)/127.5

            #train
            model.fit(X_train)

        else:
            save_animation(15, 'results/dcgan_animation.gif', 'results/dcgan_fixed_results/')
            model.load(args.restoring_epoch)
            if args.model == 0:
                img_path = 'results/gan_genarated_image.png'
                img_size = (28, 28)
            elif args.model == 1:
                img_path = 'results/dcgan_genarated_image.png'
                img_size = (64, 64)
            save_results(model, args.restoring_epoch, img_path, dim = (7, 7), figsize=(7, 7), img_size =img_size)