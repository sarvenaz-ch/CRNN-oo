# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 12:10:23 2021

@author: local_ergo
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def sliding_window_numbers(n_samples, sequence_length, shift):
    ''' This function takes in the total number of samples we have (n_samples)
    the length of the windows we want to have (sequence_length) and how many
    steps each window slides/shifts (shift) and returns number of windows 
    possible to create in that dataset and number of remainder items in n_samples
    that cannot be covered with the windows
    '''
    r = n_samples
    n_windows = 0 # number of sliding windows
    while r > sequence_length:
        n_windows += 1
        r = n_samples-n_windows*(shift)
        
    r = n_samples-(n_windows-1)*(shift) - sequence_length
    # print('n_windows:', n_windows, 'remainder:', r)
    
    return n_windows, r
    
def sliding_window_image(original_data, n_windows, sequence_length, shift, plot_samples = False, max_sensal = 302740):
    ''' This function takes in the original image data (original_data), number
    of windows (n_windows) the sequence length, number of skipping steps for
    the sliding windows (shft) and creates a 4 dimensional datasets in the form
    of: data.shape = [n_windows, seq_length, width,height] for RNN
    '''
    
    new_data = np.zeros((n_windows, sequence_length, original_data.shape[1], original_data.shape[2]))
    cc = 0 # counter
    for i in range(n_windows):    
        new_data[i]= original_data[cc+0:cc+sequence_length][:][:]
        cc += shift
        
    if plot_samples == True:
        print('- Plotting a few samples for the new image data...')
        for i in range(5):
            im = Image.fromarray(np.uint8(max_sensal*new_data[0,i,:,:]),'L')
            plt.imshow(im, cmap="gray") 
            plt.show()
    return new_data

def sliding_window_2outputs(original_data, n_windows, sequence_length, shift):
    new_data = np.zeros((n_windows, sequence_length, original_data.shape[1]))
    cc = 0 # counter
    for i in range(n_windows):    
        new_data[i]= original_data[cc+0:cc+sequence_length][:]
        cc += shift
    return new_data