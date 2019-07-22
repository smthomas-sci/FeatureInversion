"""
Useful functions specific for VGG16
"""
from skimage.transform import resize
from skimage.io import imread, imsave

import numpy as np

import os

def preprocess_image(x):
    # convert to float
    x = x.astype("float32")
    # zero-center by mean pixel
    x[:, :, 0] -= 123.68
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 103.939
    # 'RGB' -> 'BGR'
    x = x[:, :, ::-1]
    return x[..., 0:3]
    
def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
    
def total_variation_loss(x, img_nrows=224,img_ncols=224, beta=2):
    a = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, beta))


class Generator(object):
    """
    A generator class that grabs and image and creates a batch
    where X == y
    """
    
    def __init__(self, files, batch_size = 6):
        self.files = files
        self.batch_size = batch_size
        self.i = 0
        self.n = len(self.files)
        self.order = np.arange(self.n)
        np.random.shuffle(self.order)
        
        
    def __next__(self):
        batch = []
        
        for step in range(self.batch_size):
            # Check for wrapping
            if self.i >= len(self.files):
                # Start again
                self.i = 0
                # Reshuffle
                np.random.shuffle(self.order)
                         
            x = preprocess_image(imread(self.files[self.order[self.i]]))
            
            # resize
            x = resize(x, (224, 224))
            
            batch.append(x)
            
            self.i += 1
        
        
        return np.stack(batch), np.stack(batch)
        
