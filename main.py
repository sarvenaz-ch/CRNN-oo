# -*- coding: utf-8 -*-
"""
This main file is built on main_v01 . This main file train networks with multiple
subject data. The data are stored in folder dataset and are not synced with the 
corresponding remote repository on GitHub. Any changes made to the dataset should
be synced manually
@author: local_ergo
"""
import os
import pickle
import time
import numpy as np
import keras
from PIL import Image
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import img_to_array, load_img
# from keras.callbacks import ModelCheckpoint

# from functions_create_models import create_crnn_model, model_clstm_2outputs_batch_normalized
# from functions_preprocessing import sliding_window_numbers, sliding_window_image, sliding_window_2outputs
# from functions_visualization import plot_training_and_validation, plot_validation, plot_training
from functions_train import train
from classes_Dataset import Multi_Datasets
from classes_Models import ResearchModels


if __name__ == "__main__":
    #-----------------------------------------------------------------------------
    #                           USER INPUT
    #-----------------------------------------------------------------------------
    '''model parameter'''
    model_type = 'lrcn' # The model. For more info, refer to classes_Models.py
    num_ch = 1 # number of color channels (1 -> grey scale)
    seq_length = 70 # length of sequence of data for each pocket of data for LSTM
    shift = 10
    val_size = 0.2 # 0 if no validation set
    loss = keras.losses.MeanSquaredError()
    subject_n = [[1,1],[4,1],[7,1],[8,1],[8,2],[10,1],[10,2],[12,1],[13,1],[13,2]] #subject_n = [[subject number,trial number]]
    
    epochs = 10 # number of epochs for each batch of training set
    n_splits= 5 # k-fold cross validation parameter, number of folds
    drop_out = 0
    batch_normalization = True
    
    num_outputs = 2 # number of outputs (6 for all components, 4 for fx's and fy's, 2 for fz's)    
    
    '''visualization'''
    model_summary = True  # Show the summary of the model
    training_plot = True  # Show the training plot
    validation_plot = True # Show the validation plot
    training_and_validation_plot = True # show the training and validation on top pf each other
    
    n_skip_training = 0 # number of data points to skip for plotting the training data
    n_skip_validation = 0 # number of data points to skip for plotting the validation data

    #__________________________________________________________________________
    #__________________________________________________________________________
    
    start_time = time.time()
    
    '''DATA'''
    data = Multi_Datasets(model = model_type, subject_n = subject_n, 
                          seq_length = seq_length, shift = shift, test_size = 0.2, val_size = val_size)
    for model_type in ['cnn', 'lrcn', 'c3d']:
        ''' THE MODEL'''
        model = ResearchModels(model = model_type, seq_length = seq_length,
                               image_size = data.image_size, num_ch = num_ch,
                               saved_model=None, model_summary = True)
    #%%
        '''TRAINING'''
        if model_type == 'cnn':
            data.train_x = np.array([data.train_x.T]).T # adding a dimension for channel
            history = train(model = model.model, train_x = data.train_x, train_label = data.train_label,
                epochs = epochs, verbose = 1, train_plot = False)
        elif model_type in ['lrcn', 'c3d']:
            data.chopped_train_x =  np.array([data.chopped_train_x.T]).T # adding a dimension for channel
            history = train(model = model.model, train_x = data.chopped_train_x, train_label = data.chopped_train_label,
                    epochs = epochs, verbose = 1, train_plot = False)
        # train(model = model.model, train_x = data.chopped_train_x, train_label = data.chopped_train_label,
        #       val_x = data.chopped_val_x, val_label = data.chopped_val_label,
        #       epochs = epochs, verbose = 1, train_plot = False)
        ''' VISUALIZATION'''
        plt.plot(history['loss'], label = 'Loss')
        plt.plot(history['fz1_loss'], label = 'fz1 MSE Loss')
        plt.plot(history['fz2_loss'], label = 'fz2 MSE Loss')
        plt.title('The loss value for '+model_type)
        plt.xlabel('epoch'); plt.ylabel('MSE'); plt.legend(); plt.show()
        plt.plot(history['fz1_root_mean_squared_error'], label = 'fz1 RMSE Loss')
        plt.plot(history['fz2_root_mean_squared_error'], label = 'fz2 RMSE Loss')
        plt.title('The loss value for '+model_type)
        plt.xlabel('epoch'); plt.ylabel('RMSE'); plt.legend(); plt.show()