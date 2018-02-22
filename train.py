from gan import GAN
import numpy as np
import argparse
from utils import create_results_dir, save_results


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--keep_prob', type=float, default=0.3)
    parser.add_argument('--restoring_epoch', type=int, default=500)
    parser.add_argument('--random_dim', type=int, default=100)
    args = parser.parse_args(*argument_array)
    return args

if __name__ == "__main__":

    args = parse_args()

    gan = GAN(epochs = args.epochs, 
              lr_rate = args.lr_rate,
              batch_size = args.batch_size, 
              keep_prob = args.keep_prob,
              random_dim = args.random_dim)

    if args.train == 'True':
        create_results_dir()

        # load MNIST
        X_train = np.load('MNIST_data/mnist_train_x.npy')
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train.reshape(60000, 784)


        # train
        gan.fit(X_train)

    else:
        gan.load(args.restoring_epoch)
        save_results(gan, args.random_dim, args.restoring_epoch, 'results/gan_genarated_image.png')