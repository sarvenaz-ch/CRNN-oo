"""
Created on Sat Apr  3 12:09:15 2021

@author: local_ergo
"""
import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from functions_preprocessing import sliding_window_numbers, sliding_window_image, sliding_window_2outputs

class Dataset():
    def __init__(self, model, sensal_filename, FP_filename, seq_length = 30, shift = 10,
                 test_size = 0.2, val_size = 0):
        print('- Loading the data ...')
        sensal = pickle.load(open(sensal_filename, 'rb')); sensal = sensal['sensal']
        FP = pickle.load(open(FP_filename, 'rb'))
        sensal = sensal[50:44200][:][:]
        
        self.FP = FP[50:44200][:]
        temp1 = np.amax(self.FP[:,5])
        temp2 = np.amax(self.FP[:,2])
        self.max_FP = max(temp1, temp2)
        del temp1, temp2
        for i in range(self.FP.shape[1]):
            self.FP[:,i] = (self.FP[:,i]- np.amin(self.FP[:,i]))/(np.amax(self.FP[:,i])- np.amin(self.FP[:,i]))

        
        #----------------- VISUALIZATION
        # im = Image.fromarray(np.uint8(sensal[2,:,:]),'L')
        # plt.imshow(im, cmap="magma") 
        # plt.title('A sample of the input data')
        # plt.show()
        # plt_fz1, = plt.plot(FP[:,2])
        # plt_fz2, = plt.plot(FP[:,5])
        # plt.title('Fz1 and Fz2');
        # plt.legend((plt_fz1,plt_fz2), ('fz1', 'fz2'),loc='upper right')
        # plt.show()
        # del plt_fz1, plt_fz2
        
        #------------ INPUT DATA NORMALIZATION TO BE BETWEEN (0,1)
        self.max_sensal = np.amax((sensal))
        self.sensal = sensal/self.max_sensal
        # self.sensal = np.expand_dims(self.sensal, axis = 3)
        # print('shape:',self.sensal.shape)
        #----------------- TRAIN AND TEST DATASET ---------------------------------------- 

        if model in ['lrcn', 'convlstm2d', 'c3d']:
            print('- Creating test and train datasets for', model,' model...')
            self.train_x, self.test_x, train_output, test_label = train_test_split(self.sensal, self.FP,
                                                                  test_size = test_size,
                                                                  random_state = 13)
            # Components of the force                                                              
            fz1_train_label = train_output[:,2]
            fz2_train_label = train_output[:,5]
            self.train_label = [fz1_train_label, fz2_train_label]
            
            fz1_test_label = test_label[:,2]
            fz2_test_label = test_label[:,5]
            del test_label
            self.test_label = np.transpose(np.array([fz1_test_label, fz2_test_label]))
            
            # If we have validation set
            if val_size > 0:
                print('- Creating validation and train datasets...')
                self.train_x, self.val_x, train_label, val_label = train_test_split(self.train_x, train_output,
                                                                          test_size = val_size,
                                                                          random_state = 13)
                # Components of the force                                                              
                fz1_train_label = train_label[:,0]
                fz2_train_label = train_label[:,1]
                train_output = []; train_output.append(fz1_train_label);train_output.append(fz2_train_label)
                fz1_val_label = val_label[:,0]
                fz2_val_label = val_label[:,1]
                val_output = []; val_output.append(fz1_val_label); val_output.append(fz2_val_label);
                del val_label, train_label
                self.train_label = np.transpose(np.array(train_output))
                self.val_label = np.transpose(np.array(val_output))
            else:
                self.train_label = np.transpose(np.array(self.train_label))
            
            
            print('- Chopping the original data to sequeces of length of ', seq_length,'...')
            self.image_size = (self.train_x.shape[1], self.train_x.shape[2]) # The size of the image matrix at each frame  
            assert len(self.train_x) >= seq_length
            
            '''Chopping Train Data'''
            n_frames = len(self.train_x) # number of frames
            n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
            self.chopped_train_x = sliding_window_image(original_data = self.train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False, max_sensal=self.max_sensal)
            self.chopped_train_label = sliding_window_2outputs(original_data = self.train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
            '''Chopping Test Data'''
            n_frames = len(self.test_x) # number of frames
            n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
            self.chopped_test_x = sliding_window_image(original_data = self.test_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
            self.chopped_test_label = sliding_window_2outputs(original_data = self.test_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
            if val_size > 0:
                '''Chopping Validation Data'''
                n_frames = len(self.val_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                self.chopped_val_x = sliding_window_image(original_data = self.val_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                self.chopped_val_label = sliding_window_2outputs(original_data = self.val_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
       
        #----------------------------------------
        # elif model == 'c3d':
        #     print('- Creating test and train datasets for c3d model...')
        #     self.train_x, self.test_x, train_output, test_label = train_test_split(self.sensal, self.FP,
        #                                                           test_size = test_size,
        #                                                           random_state = 13)
        #     # Components of the force                                                              
        #     fz1_train_label = train_output[:,2]
        #     fz2_train_label = train_output[:,5]
        #     self.train_label = [fz1_train_label, fz2_train_label]
            
        #     fz1_test_label = test_label[:,2]
        #     fz2_test_label = test_label[:,5]
        #     del test_label
        #     self.test_label = np.transpose(np.array([fz1_test_label, fz2_test_label]))
            
        #     # If we have validation set
        #     if val_size > 0:
        #         print('- Creating validation and train datasets...')
        #         self.train_x, self.val_x, train_label, val_label = train_test_split(self.train_x, train_output,
        #                                                                   test_size = val_size,
        #                                                                   random_state = 13)
        #         # Components of the force                                                              
        #         fz1_train_label = train_label[:,0]
        #         fz2_train_label = train_label[:,1]
        #         train_output = []; train_output.append(fz1_train_label);train_output.append(fz2_train_label)
        #         fz1_val_label = val_label[:,0]
        #         fz2_val_label = val_label[:,1]
        #         val_output = []; val_output.append(fz1_val_label); val_output.append(fz2_val_label);
        #         del val_label, train_label
        #         self.train_label = np.transpose(np.array(train_output))
        #         self.val_label = np.transpose(np.array(val_output))
        #     else:
        #         self.train_label = np.transpose(np.array(self.train_label))
            
            
        #     print('- Chopping the original data to sequeces of length of ', seq_length,'...')
        #     self.image_size = (self.train_x.shape[1], self.train_x.shape[2]) # The size of the image matrix at each frame  
        #     assert len(self.train_x) >= seq_length
            
        #     '''Chopping Train Data'''
        #     n_frames = len(self.train_x) # number of frames
        #     n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
        #     self.chopped_train_x = sliding_window_image(original_data = self.train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False, max_sensal=self.max_sensal)
        #     self.chopped_train_label = sliding_window_2outputs(original_data = self.train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
        #     '''Chopping Test Data'''
        #     n_frames = len(self.test_x) # number of frames
        #     n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
        #     self.chopped_test_x = sliding_window_image(original_data = self.test_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
        #     self.chopped_test_label = sliding_window_2outputs(original_data = self.test_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
        #     if val_size > 0:
        #         '''Chopping Validation Data'''
        #         n_frames = len(self.val_x) # number of frames
        #         n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
        #         self.chopped_val_x = sliding_window_image(original_data = self.val_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
        #         self.chopped_val_label = sliding_window_2outputs(original_data = self.val_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
        else:
            raise ValueError('The model is not either passed to the Datasets class or is not defined.')
            


class Multi_Datasets():
    def __init__(self, model, subject_n, seq_length = 30, shift = 10,
                 test_size = 0.2, val_size = 0):
        self.model = model
        self.subject_n = subject_n
        self.seq_length = seq_length
        
        self.FPs = []
        self.sensals = []
        print('- Loading the data ...')
        for [n,t] in subject_n:
            sensal_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_sensal'
            FP_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_FP'
            sensal = pickle.load(open(sensal_filename, 'rb')); sensal = sensal['sensal']
            FP = pickle.load(open(FP_filename, 'rb'))
            self.FPs.append(FP)
            self.sensals.append(sensal)
        del FP, sensal,n, t
        #----------------- VISUALIZATION
        # print('- Visualization of the training data ...')
        # for sub in range(len(self.FPs)):
        #     plt_fz1, = plt.plot(self.FPs[sub][:,2])
        #     plt_fz2, = plt.plot(self.FPs[sub][:,5])
        #     plt.title('Fz1 and Fz2 for subject'+str(sub+1));
        #     plt.legend((plt_fz1,plt_fz2), ('fz1', 'fz2'),loc='upper right')
        #     plt.show()
        #     del plt_fz1, plt_fz2
        # im = Image.fromarray(np.uint8(self.sensals[0][15,:,:]),'L')
        # plt.imshow(im, cmap="magma") 
        # plt.title('A sample of the input data')
        # plt.show()
        
        #------------ INPUT DATA NORMALIZATION TO BE BETWEEN (0,1)
        print('- Normalizing the data ...')
        for sub in range(len(self.subject_n)):
            max_sensal = np.amax(self.sensals[sub])
            self.sensals[sub] = self.sensals[sub]/max_sensal
            temp1 = np.amax(self.FPs[sub][:,5])
            temp2 = np.amax(self.FPs[sub][:,2])
            max_FP = max(temp1, temp2)
            
            for i in range(self.FPs[sub].shape[1]):
                self.FPs[sub][:,i] = (self.FPs[sub][:,i]- np.amin(self.FPs[sub][:,i]))/(np.amax(self.FPs[sub][:,i])- np.amin(self.FPs[sub][:,i]))
            del temp1, temp2, i
        
        
        print('- Creating test and train datasets for', model,' model...')
        self.train_xs = []; self.test_xs = [] ;
        self.train_labels =[]; self.test_labels = [];
        self.val_xs = []; self.val_labels = [];
        self.chopped_train_xs = []; self.chopped_train_labels = [];
        self.chopped_test_xs = []; self.chopped_test_labels = [];
        self.chopped_val_xs = []; self.chopped_val_labels = [];
        for sub in range(len(subject_n)):    
            train_x, test_x, train_output, test_label = train_test_split(self.sensals[sub], self.FPs[sub],
                                                              test_size = test_size,
                                                              random_state = 13)
         
          
            #Components of the force                                                              
            fz1_train_label = train_output[:,2]
            fz2_train_label = train_output[:,5]
            train_label = [fz1_train_label, fz2_train_label]
            
            fz1_test_label = test_label[:,2]
            fz2_test_label = test_label[:,5]
            del test_label
            test_label = np.transpose(np.array([fz1_test_label, fz2_test_label]))
            
        
            # If we have validation set
            if val_size > 0:
    
                train_x, val_x, train_label, val_label = train_test_split(train_x, train_output,
                                                                          test_size = val_size,
                                                                          random_state = 13)
                # Components of the force                                                              
                fz1_train_label = train_label[:,2]
                fz2_train_label = train_label[:,5]
                train_output = []; train_output.append(fz1_train_label);train_output.append(fz2_train_label)
                fz1_val_label = val_label[:,2]
                fz2_val_label = val_label[:,5]
                val_output = []; val_output.append(fz1_val_label); val_output.append(fz2_val_label);
                del val_label, train_label
                train_label = np.transpose(np.array(train_output))
                val_label = np.transpose(np.array(val_output))
                self.val_xs.append(val_x)
                self.val_labels.append(val_label);
                
            else:
                train_label = np.transpose(np.array(train_label))    
                
            self.image_size = (train_x.shape[1], train_x.shape[2]) # The size of the image matrix at each frame  
            
            # Chop the data for temporal networks
            if model in ['lrcn', 'c3d', 'convlstm2d']:
#                print('- Chopping the original data to sequeces of length of ', seq_length,'...')
                assert len(train_x) >= seq_length
                
                '''Chopping Train Data'''
                n_frames = len(train_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_train_x = sliding_window_image(original_data = train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False, max_sensal=max_sensal)
                chopped_train_label = sliding_window_2outputs(original_data = train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                ''' Chopping Test Data'''
                assert len(test_x) >= seq_length
                n_frames = len(test_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_test_x = sliding_window_image(original_data = test_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                chopped_test_label = sliding_window_2outputs(original_data = test_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
                if val_size > 0:
                    '''Chopping Validation Data'''
                    n_frames = len(val_x) # number of frames
                    assert len(val_x) >= seq_length
                    n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                    chopped_val_x = sliding_window_image(original_data = val_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                    chopped_val_label = sliding_window_2outputs(original_data = val_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                    self.chopped_val_xs.append(chopped_val_x); self.chopped_val_labels.append(chopped_val_label); 
                
                self.chopped_train_xs.append(chopped_train_x); self.chopped_train_labels.append(chopped_train_label);
                self.chopped_test_xs.append(chopped_test_x); self.chopped_test_labels.append(chopped_test_label);
                del chopped_test_x, chopped_train_x, chopped_test_label, chopped_train_label
                
            self.train_xs.append(train_x); self.test_xs.append(test_x)
            self.train_labels.append(train_label); self.test_labels.append(test_label)
    
            del sub, train_x, test_x, train_output, test_label,train_label, fz1_test_label, fz2_test_label, fz1_train_label, fz2_train_label
        
        #___________ END of for loop __________________
        
        
        if val_size > 0:
            print('- This datset contains a validation dataset. To disable the validation set, assign 0 to val_size')
            self.seq_lim = min(map(len,self.val_labels)) # The maximum sequence length I can choose
        else:    
            self.seq_lim = min(map(len,self.test_labels)) # The maximum sequence length I can choose
    
        
        ''' CONCATINATING THE DATA'''
        self.train_x = np.vstack(self.train_xs); self.train_label = np.vstack(self.train_labels);
        self.test_x = np.vstack(self.test_xs); self.test_label = np.vstack(self.test_labels);
        if val_size > 0:
            self.val_x = np.vstack(self.val_xs); self.val_label = np.vstack(self.val_labels);
        if model in ['lrcn', 'c3d', 'convlstm2d']:
            self.chopped_train_x = np.vstack(self.chopped_train_xs); 
            self.chopped_train_label = np.vstack(self.chopped_train_labels);
            self.chopped_test_x = np.vstack(self.chopped_test_xs);
            self.chopped_test_label = np.vstack(self.chopped_test_labels);
            if val_size > 0:
                self.chopped_val_x = np.vstack(self.chopped_val_xs); 
                self.chopped_val_label = np.vstack(self.chopped_val_labels);
                
                
                
#------------------------ SHEAR FORCES ---------------------------------------

class Multi_Datasets_fxfy():
    def __init__(self, model, subject_n, seq_length = 30, shift = 10,
                 test_size = 0.2, val_size = 0):
        self.model = model
        self.subject_n = subject_n
        self.seq_length = seq_length
        
        self.FPs = []
        self.sensals = []
        print('- Loading the data ...')
        for [n,t] in subject_n:
            sensal_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_sensal'
            FP_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_FP'
            sensal = pickle.load(open(sensal_filename, 'rb')); sensal = sensal['sensal']
            FP = pickle.load(open(FP_filename, 'rb'))
            self.FPs.append(FP)
            self.sensals.append(sensal)
        del FP, sensal,n, t
        #----------------- VISUALIZATION
        print('- Visualization of the training data ...')
        for sub in range(len(self.FPs)):
            plt.plot(self.FPs[sub][:,0], label='fx1')
            plt.plot(self.FPs[sub][:,1], label='fy1')
            plt.plot(self.FPs[sub][:,3], label='fx2')
            plt.plot(self.FPs[sub][:,4], label='fy2')
            plt.title('Shear forces for subject'+str(sub+1));
            plt.legend(loc='upper right')
            plt.show()
            
        im = Image.fromarray(np.uint8(self.sensals[0][15,:,:]),'L')
        plt.imshow(im, cmap="magma") 
        plt.title('A sample of the input data')
        plt.show()
        
        #------------ INPUT DATA NORMALIZATION TO BE BETWEEN (0,1)
        print('- Normalizing the data ...')
        for sub in range(len(self.subject_n)):
            max_sensal = np.amax(self.sensals[sub])
            self.sensals[sub] = self.sensals[sub]/max_sensal
            temp1 = np.amax(self.FPs[sub][:,0])
            temp2 = np.amax(self.FPs[sub][:,1])
            temp3 = np.amax(self.FPs[sub][:,3])
            temp4 = np.amax(self.FPs[sub][:,4])
            max_FP = max(temp1, temp2, temp3, temp4) #find the maximum value of shear force (over x and y)
            
            for i in range(self.FPs[sub].shape[1]):
                self.FPs[sub][:,i] = (self.FPs[sub][:,i]- np.amin(self.FPs[sub][:,i]))/(np.amax(self.FPs[sub][:,i])- np.amin(self.FPs[sub][:,i]))
            del temp1, temp2, temp3, temp4, i
        
        
        print('- Creating test and train datasets for', model,' model...')
        self.train_xs = []; self.test_xs = [] ;
        self.train_labels =[]; self.test_labels = [];
        self.val_xs = []; self.val_labels = [];
        self.chopped_train_xs = []; self.chopped_train_labels = [];
        self.chopped_test_xs = []; self.chopped_test_labels = [];
        self.chopped_val_xs = []; self.chopped_val_labels = [];
        for sub in range(len(subject_n)):    
            train_x, test_x, train_output, test_label = train_test_split(self.sensals[sub], self.FPs[sub],
                                                              test_size = test_size,
                                                              random_state = 13)
         
          
            #Components of the force                                                              
            fx1_train_label = train_output[:,0]
            fy1_train_label = train_output[:,1]
            fx2_train_label = train_output[:,3]
            fy2_train_label = train_output[:,4]
            train_label = [fx1_train_label, fy1_train_label,fx2_train_label, fy2_train_label]
            
            fx1_test_label = test_label[:,0]
            fy1_test_label = test_label[:,1]
            fx2_test_label = test_label[:,3]
            fy2_test_label = test_label[:,4]
            del test_label
            test_label = np.transpose(np.array([fx1_test_label, fy1_test_label,
                                                fx2_test_label, fy2_test_label]))
                 
            # If we have validation set
            if val_size > 0:
                train_x, val_x, train_label, val_label = train_test_split(train_x, train_output,
                                                                          test_size = val_size,
                                                                          random_state = 13)
                # Components of the force                                                              
                fx1_train_label = train_label[:,0]
                fy1_train_label = train_label[:,1]
                fx2_train_label = train_label[:,3]
                fy2_train_label = train_label[:,4]
                train_output = []
                train_output.append(fx1_train_label);train_output.append(fy1_train_label)
                train_output.append(fx2_train_label);train_output.append(fy2_train_label)
                fx1_val_label = val_label[:,0]
                fy1_val_label = val_label[:,1]
                fx2_val_label = val_label[:,3]
                fy2_val_label = val_label[:,4]
                val_output = []
                val_output.append(fx1_val_label); val_output.append(fy1_val_label);
                val_output.append(fx2_val_label); val_output.append(fy2_val_label);
                del val_label, train_label
                train_label = np.transpose(np.array(train_output))
                val_label = np.transpose(np.array(val_output))
                self.val_xs.append(val_x) 
                self.val_labels.append(val_label); 
                
            else:
                train_label = np.transpose(np.array(train_label))    
                
            self.image_size = (train_x.shape[1], train_x.shape[2]) # The size of the image matrix at each frame  
            
            # Chop the data for temporal networks
            if model in ['lrcn', 'c3d', 'convlstm2d']:
#                print('- Chopping the original data to sequeces of length of ', seq_length,'...')
                assert len(train_x) >= seq_length
                
                '''Chopping Train Data'''
                n_frames = len(train_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_train_x = sliding_window_image(original_data = train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False, max_sensal=max_sensal)
                chopped_train_label = sliding_window_2outputs(original_data = train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                ''' Chopping Test Data'''
                assert len(test_x) >= seq_length
                n_frames = len(test_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_test_x = sliding_window_image(original_data = test_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                chopped_test_label = sliding_window_2outputs(original_data = test_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
                if val_size > 0:
                    '''Chopping Validation Data'''
                    n_frames = len(val_x) # number of frames
                    assert len(val_x) >= seq_length
                    n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                    chopped_val_x = sliding_window_image(original_data = val_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                    chopped_val_label = sliding_window_2outputs(original_data = val_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                    self.chopped_val_xs.append(chopped_val_x); self.chopped_val_labels.append(chopped_val_label); 
                
                self.chopped_train_xs.append(chopped_train_x); self.chopped_train_labels.append(chopped_train_label);
                self.chopped_test_xs.append(chopped_test_x); self.chopped_test_labels.append(chopped_test_label);
                del chopped_test_x, chopped_train_x, chopped_test_label, chopped_train_label
                
            self.train_xs.append(train_x); self.test_xs.append(test_x)
            self.train_labels.append(train_label); self.test_labels.append(test_label)
    
            del sub, train_x, test_x, train_output, test_label,train_label,fx1_test_label, fx2_test_label,fy1_test_label, fy2_test_label, fx1_train_label, fx2_train_label, fy1_train_label, fy2_train_label
        
        #___________ END of for loop __________________
        
        
        if val_size > 0:
            print('- This datset contains a validation dataset. To disable the validation set, assign 0 to val_size')
            self.seq_lim = min(map(len,self.val_labels)) # The maximum sequence length I can choose
        else:    
            self.seq_lim = min(map(len,self.test_labels)) # The maximum sequence length I can choose
    
        
        ''' CONCATINATING THE DATA'''
        self.train_x = np.vstack(self.train_xs); self.train_label = np.vstack(self.train_labels);
        self.test_x = np.vstack(self.test_xs); self.test_label = np.vstack(self.test_labels);
        if val_size > 0:
            self.val_x = np.vstack(self.val_xs); self.val_label = np.vstack(self.val_labels);
        if model in ['lrcn', 'c3d', 'convlstm2d']:
            self.chopped_train_x = np.vstack(self.chopped_train_xs); 
            self.chopped_train_label = np.vstack(self.chopped_train_labels);
            self.chopped_test_x = np.vstack(self.chopped_test_xs);
            self.chopped_test_label = np.vstack(self.chopped_test_labels);
            if val_size > 0:
                self.chopped_val_x = np.vstack(self.chopped_val_xs); 
                self.chopped_val_label = np.vstack(self.chopped_val_labels);