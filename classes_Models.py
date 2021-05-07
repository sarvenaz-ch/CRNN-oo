# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:29:40 2021

@author: local_ergo
"""
import keras
from keras.layers import (Dense, Flatten, Dropout, ZeroPadding3D, Input,
                          Activation, BatchNormalization, ConvLSTM2D)
from keras.losses import MeanSquaredError
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop, Adagrad
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
from keras.initializers import he_uniform, glorot_uniform, RandomUniform
import sys


class ResearchModels():
    def __init__(self, model = 'cnn', seq_length = 0, image_size = [32,22],
                 num_ch = 1, num_output = 2, 
                 cnn_kernel_initializars =  glorot_uniform(), lr = 1e-4,
                 lstm_kernel_initializars =  glorot_uniform(),
                 dense_kernel_initializars =  glorot_uniform(), saved_model=None,
                 features_length=2048, model_summary = False):
        """
        `model` = one of:
            cnn
            lstm
            lrcn
            convlstm2d
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.image_size = image_size
        self.num_ch = num_ch
        self.num_output = num_output
        self.load_model = load_model
        self.saved_model = saved_model
        self.feature_queue = deque()
        self.model_type = model
        self.kernel_init_cnn = cnn_kernel_initializars
        self.kernel_init_lstm = lstm_kernel_initializars
        self.kernel_init_dense = dense_kernel_initializars


        # Get the appropriate model.
        if self.saved_model is not None:
            print("- Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'cnn':
            print("- Loading CNN model...")
            self.input_shape = (image_size[0], image_size[1], num_ch)
            self.model = self.cnn()
        elif model == 'lstm':
            print("- Loading LSTM model...")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("- Loading Long-Term Recurrent Convolution Network ...")
            print('seq_length:', seq_length, "image_size[0]:", image_size[0],"image_size[1]:", image_size[1], "num_ch:", num_ch)
            self.input_shape = (seq_length, image_size[0], image_size[1], num_ch)
            self.model = self.lrcn()
        elif model == 'c3d':
            print('- Loading C3D ...')
            self.input_shape = (seq_length, image_size[0], image_size[1], num_ch)
            self.model = self.c3d()
        elif model == 'convlstm2d':
            print('- Loading Convolutional LSTM (convlstm2d) model ...')
            self.input_shape = (seq_length, image_size[0], image_size[1], num_ch)
            self.model = self.convlstm2d()
        else:
            raise ValueError("- Unknown network!!!. See classes_Models.py for the list of valid models")
            sys.exit()
                
        if model_summary == True:
            self.model.summary()    
        
        # Now compile the network.
#        optimizer = Adam(lr=1e-5, decay=1e-6)

        optimizer = Adam(learning_rate = lr)
        self.model.compile(loss = keras.losses.MeanSquaredError(reduction="none"),
                  optimizer = Adagrad(), metrics=[keras.metrics.RootMeanSquaredError()])


    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                        input_shape=self.input_shape,
                        dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def cnn(self):
        ''' This function trains a model for 2 outputs, fz's of two force plates'''
        input_layer = Input(shape = self.input_shape)
        _ = Conv2D(32, kernel_size = (3,3), activation = 'relu',
                   input_shape = (31,22,1), padding = 'same' )(input_layer)
        _ = MaxPooling2D((2,2), padding = 'same')(_)
        _ = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(_)
        _ = MaxPooling2D((2,2), padding = 'same')(_)
        _ = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(_)
        _ = MaxPooling2D((2,2), padding = 'same')(_)
        last_common_layer = Flatten()(_)
        if self.num_output ==2:
            #------ FP 1 output'''   
            _ = Dense(units=128, activation='relu')(last_common_layer)
            _ = Dense(units=64, activation='relu')(_)
            _ = Dense(units=32, activation='relu')(_)
            fz1_output = Dense(units=1, activation='relu', name='fz1')(_)
                
            #------FP 2 output'''
            _ = Dense(units=128, activation='relu')(last_common_layer)
            _ = Dense(units=64, activation='relu')(_)
            _ = Dense(units=32, activation='relu')(_)
            fz2_output = Dense(units=1, activation='relu', name='fz2')(_)
                        
            outputs = [fz1_output, fz2_output]
            
        elif self.num_output == 4:
            _ = Dense(units=128, activation='relu', name = 'dense_128_fx1')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fx1')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fx1')(_)
            fx1_output = Dense(units=1, activation='relu', name='fx1')(_)
            
            _ = Dense(units=128, activation='relu', name = 'dense_128_fy1')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fy1')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fy1')(_)
            fy1_output = Dense(units=1, activation='relu', name='fy1')(_)
        
            _ = Dense(units=128, activation='relu', name = 'dense_128_fx2')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fx2')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fx2')(_)
            fx2_output = Dense(units=1, activation='relu', name='fx2')(_)
            
            _ = Dense(units=128, activation='relu', name = 'dense_128_fy2')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fy2')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fy2')(_)
            fy2_output = Dense(units=1, activation='relu', name='fy2')(_)
            
            outputs = [fx1_output, fy1_output, fx2_output, fy2_output]
        else:
            raise ValueError(' Invalid number of input is passed to the class model!')
        
        model = Model(inputs=input_layer, outputs=outputs)
        
        # plot_model(cnn_model, to_file=cnn_model_name+'.png',
        #            show_shapes=True, show_layer_names=True)
        return model

    def lrcn(self):
        '''
        This function creates a CRNN network based on the work presented in this
        paper:
            https://www.hindawi.com/journals/complexity/2020/3536572/
        This network consists of three blocks of CNN and one LSTM layer following
        by a dense layer that outputs two outputs: fz1 and fz2
        INPUTS:
        model name: is the name passed by the user under which the model saves
        time_stamp: number of instances for each time sequence
        num_ch = number of color channels of the image
        '''
        # input_layer = Input(shape=(sequence_length, image_size[0]* image_size[1]* num_color_channel))
        # input_layer = Input(shape = (self.seq_length, image_size[0],image_size[1], num_ch))
        input_layer = Input(shape = self.input_shape)
        
        #1st CNN block
        _ = TimeDistributed(Conv2D(32, kernel_size = (3,3), padding = 'same',
                                   kernel_initializer= self.kernel_init_cnn,
                                   input_shape = (self.image_size[0], self.image_size[1], self.num_ch)), name = 'Conv2D_1')(input_layer)
        _ = BatchNormalization()(_)
        _ = Activation('relu', name = 'ReLu1')(_)
        _ = TimeDistributed(MaxPooling2D((2,2), padding = 'same'), name = 'MaxPool_1')(_)
        #2nd CNN block
        _ = TimeDistributed(Conv2D(64, kernel_size = (3,3), padding = 'same', 
                                   kernel_initializer= self.kernel_init_cnn,
                                   input_shape = (self.image_size[0], self.image_size[1], self.num_ch)), name = 'Conv2D_2')(_)
        _ = BatchNormalization()(_)
        _ = Activation('relu', name = 'ReLu2')(_)
        _ = TimeDistributed(MaxPooling2D((2,2), padding = 'same'), name = 'MaxPool_2')(_)
        #3rd CNN block
        _ = TimeDistributed(Conv2D(128, kernel_size = (3,3), padding = 'same',
                                   kernel_initializer= self.kernel_init_cnn,
                                   input_shape = (self.image_size[0], self.image_size[1], self.num_ch)), name = 'Conv2D_3')(_)
        _ = BatchNormalization()(_)
        _ = Activation('relu', name = 'ReLu3')(_)
        _ = TimeDistributed(MaxPooling2D((2,2), padding = 'same'), name = 'MaxPool_3')(_)
        #LSTM
        _ = TimeDistributed((Flatten()), name = 'Flatten')(_)
        last_common_layer = LSTM(units = 32, activation='tanh', return_sequences=True, 
                                 kernel_initializer= self.kernel_init_lstm, dropout = 0.75, name = 'LSTM')(_)
        if self.num_output == 2:
            #------ FP 1 output'''   
            _ = Dense(units=128, activation='relu', name = 'dense_128_fz1')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fz1')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fz1')(_)
            fz1_output = Dense(units=1, activation='relu', name='fz1')(_)
            
            #------FP 2 output'''
            _ = Dense(units=128, activation='relu', name = 'dense_128_fz2')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fz2')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fz2')(_)
            fz2_output = Dense(units=1, activation='relu', name='fz2')(_)
            
            outputs = [fz1_output, fz2_output]
        
        elif self.num_output == 4:
            _ = Dense(units=128, activation='relu', name = 'dense_128_fx1')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fx1')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fx1')(_)
            fx1_output = Dense(units=1, activation='relu', name='fx1')(_)
            
            _ = Dense(units=128, activation='relu', name = 'dense_128_fy1')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fy1')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fy1')(_)
            fy1_output = Dense(units=1, activation='relu', name='fy1')(_)
        
            _ = Dense(units=128, activation='relu', name = 'dense_128_fx2')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fx2')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fx2')(_)
            fx2_output = Dense(units=1, activation='relu', name='fx2')(_)
            
            _ = Dense(units=128, activation='relu', name = 'dense_128_fy2')(last_common_layer)
            _ = Dense(units=64, activation='relu', name = 'dense_64_fy2')(_)
            _ = Dense(units=32, activation='relu', name = 'dense_32_fy2')(_)
            fy2_output = Dense(units=1, activation='relu', name='fy2')(_)
            
            outputs = [fx1_output, fy1_output, fx2_output, fy2_output]
        else:
            raise ValueError(' Invalid number of input is passed to the class model!')
        

        
        model = Model(inputs=input_layer, outputs=outputs)
        
        return model
    
    def convlstm2d(self):
        input_layer = Input(shape = self.input_shape)
        
        _ = ConvLSTM2D(filters = 32, kernel_size = (3,3),
                       padding = 'same', return_sequences = True)(input_layer)
        _ = BatchNormalization()(_)
        _ = ConvLSTM2D(filters = 64, kernel_size = (3,3),
                       padding = 'same', return_sequences = True)(_)
        _ = BatchNormalization()(_)
        _ = ConvLSTM2D(filters = 32, kernel_size = (2,2),
                       padding = 'same', return_sequences = True)(_)
        _ = BatchNormalization()(_)
        bottleneck = TimeDistributed(Flatten())(_)
        
        #------ FP 1 output'''   
        _ = Dense(units=32, activation='relu')(bottleneck)
        _ = Dense(units=16, activation='relu')(_)
        fz1_output = Dense(units=1, activation='relu', name='fz1')(_)
            
        #------FP 2 output'''
        _ = Dense(units=32, activation='relu')(bottleneck)
        _ = Dense(units=16, activation='relu')(_)
        fz2_output = Dense(units=1, activation='relu', name='fz2')(_)
        
        
        outputs = [fz1_output, fz2_output]
        
        model = Model(inputs=input_layer, outputs=outputs)
        
        return model

    
    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        input_layer = Input(shape = self.input_shape)
        print(self.input_shape)
        # 1st layer group
        _ = Conv3D(filters = 64, kernel_size = (3,3,3), activation='relu',
                         padding='same',
                         input_shape=self.input_shape, name = 'conv3D_1_64')(input_layer)
        # _ = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 1, 1),
        #                         padding='same', name='pool1')(_)
        # # 2nd layer group
        # _ = Conv3D(filters = 128, kernel_size = (3,3,3), activation='relu')(_)
        # _ = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
        #                         padding='same', name='pool2')(_)
        # 3rd layer group
        # _ = Conv3D(filters = 256, kernel_size = (3,3,3), activation='relu',
        #                   padding='same', name='conv3D_3a_256')(_)
        # _ = Conv3D(filters = 256, kernel_size = (3,3,3), activation='relu',
        #                   padding = 'same', name='conv3D_3b_256')(_)
        # _ = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
        #                         padding='same', name='pool3')(_)
        # # 4th layer group
        # _ = Conv3D(filters = 512, kernel_size = (3,3,3), activation='relu',
        #                  padding ='same', name='conv3D_4a_512')(_)
        # _ = Conv3D(filters = 512, kernel_size = (3,3,3), activation='relu',
        #                  padding ='same', name='conv3D_4b_512')(_)
        # _ = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
        #                        padding ='same', name='pool4')(_)

        # # 5th layer group
        # _ = Conv3D(filters = 512, kernel_size = (3,3,3), activation='relu',
        #                  padding='same', name='conv3D_5a_512')(_)
        # _ = Conv3D(filters = 512, kernel_size = (3,3,3), activation='relu',
        #                  padding ='same', name='conv3D_5b_512')(_)
        # _ = ZeroPadding3D(padding=(0, 1, 1))(_)
        # _ = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
        #                        padding ='valid', name='pool5')(_)
        bottleneck = TimeDistributed(Flatten())(_)

        # Fully connected layers        
        #------ FP 1 output'''   
        _ = Dense(units=128, activation='relu')(bottleneck)
        _ = Dense(units=64, activation='relu')(_)
        _ = Dense(units=32, activation='relu')(_)
        fz1_output = Dense(units=1, activation='relu', name='fz1')(_)
            
        #------FP 2 output'''
        _ = Dense(units=128, activation='relu')(bottleneck)
        _ = Dense(units=64, activation='relu')(_)
        _ = Dense(units=32, activation='relu')(_)
        fz2_output = Dense(units=1, activation='relu', name='fz2')(_)
        
        
        outputs = [fz1_output, fz2_output]
        
        model = Model(inputs=input_layer, outputs=outputs)

        return model


        return model
