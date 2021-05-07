# -*- coding: utf-8 -*-
"""
This file contains functions that work with the object oriented format of the 
DL algorithm
"""
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras
from keras import callbacks

def train(model, train_x, train_label, epochs, val_x = [], val_label = [], verbose = 1, train_plot = False):
#    tb = callbacks.TensorBoard(histogram_freq=1, write_grads = True)
    print('- Training model...' )
    if val_x == []:
        history = model.fit(train_x, train_label, verbose = verbose, epochs=epochs).history
    else:
        # history = model.fit(train_x, train_label, validation_data = (val_x, val_label),
        #                     verbose = verbose, epochs=epochs, callbacks=[PlotLossesKeras()]).history
        history = model.fit(train_x, train_label, validation_data = (val_x, val_label),
                            verbose = verbose, epochs=epochs).history
        
    # plt_loss = plt.plot(history['loss'][2:], label = 'overall loss')
    if train_plot == True:
        plt_fz1 = plt.plot(history['fz1_loss'][80:], label = 'fz1 loss')
        plt_fz2 = plt.plot(history['fz2_loss'][80:], label = 'fz2 loss')
        # plt_fz1 = plt.plot(history_bn['fz1_loss'][2:], label = 'fz1 loss after activation')
        # plt_fz2 = plt.plot(history_bn['fz2_loss'][2:], label = 'fz2 loss after activation')
        plt.legend(loc='upper right'); plt.title('Training loss')
        plt.show()
        # history, results = train_crnn_model(train_x = chopped_train_x, outputs_train = chopped_train_label,
        #                   num_outputs = 2, model = crnn_model, n_splits = n_splits,
        #                 epochs = epochs, callback_file_path = callback_file_path)
    return history