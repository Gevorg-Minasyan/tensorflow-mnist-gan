import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)


def create_results_dir():
    if not os.path.isdir('models'):
        os.mkdir('models')
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/random_results'):
        os.mkdir('results/random_results')
    if not os.path.isdir('results/fixed_results'):
        os.mkdir('results/fixed_results')



def save_results(gan, random_dim, num_epoch, path = 'result.png', isFix=False, dim=(5, 5), img_size=(28, 28), figsize=(5, 5)):
    if isFix:
        test_images = gan.generate()
    else:
        z_ = np.random.normal(0, 1, (dim[0]*dim[1], random_dim))
        test_images = gan.generate(z_)
        
    test_images = test_images.reshape(dim[0]*dim[1], img_size[0], img_size[1])
        
    fig = plt.figure(figsize=figsize)
    for i in range(test_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(test_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
   
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.close()
    

def save_train_history(history, path = 'train_history.png'):
    x = range(len(history['D_losses']))

    y1 = history['D_losses']
    y2 = history['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)
    plt.close()


def save_animation(epochs):
    images = []
    for e in range(epochs):
        img_name = 'results/fixed_results/gan_generated_image_epoch_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave('results/animation.gif', images, fps=5)


