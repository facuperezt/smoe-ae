import numpy as np
from tensorflow import keras
import tensorflow as tf
from io import BytesIO
from skimage.util import random_noise



"""
    Structure of the filebuffer FB
    Bytes 0-4: Metadata,  indicating the total number of blocks in data.
    Bytes 4-8: Length of the first data block.
    Bytes 8 - (8 + L_1): The actual first data block.
    Bytes (8 + L_1) - (12 + L_1): Length of the second data block.
    Bytes (12 + L_1) - (12 + L_1 + L_2): The actual second data block.
"""

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(16,16), noise=False,
                 shuffle=True, buffer_size=5000): #list_IDs, labels TODO no channels?
        'Initialization'
        self.dim = dim
        #self.labels = labels
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size=buffer_size
        self.fb=open(list_IDs,'rb') # hier wird filebuffer initialisiert indem list_IDs als binärer bytestream geöffnet wird (gehe davon aus, dass es sich hier um die pickles handelt.)
        self.len_data = self.open_file(self.fb) #len_data called hier die open_file Methode, welche die Länge des filebuffers zurückgibt. (also die Länge des Gesamtpickles)
        self.buffer=self.load_batch(self.buffer_size)
        self.noise=noise
        self.on_epoch_end()


    def open_file(self,fb): # Parameter fb: stands for filebuffer. 
        #The function reads the first 4 bytes from the file 
        #the first 4 bytes are then converted into an integer using Python's int.from_bytes method with little-endian byte order. In little-endian order, the least significant byte is at the smallest address
        # len_data is here the amount of 16x16 blocks that are stored in the file.
        len_data=fb.read(4)
        return int.from_bytes(len_data,'little') # the returned value is stored in self.len_data (see initialization)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.len_data / self.batch_size))#int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X,y = self.__data_generation() #X, y  #list_IDs_temp
        #y=X
        return X, y

    def load_batch(self,batch_size):
        '''
        This function loads one batch of data from the filebuffer. It reads the next 4 bytes (length of next block), advances 4 bytes, then loads the data, advances by len_ bytes, deserialize data and appends the block.
        It does batch_size iterations, in each iteration a block 16x16 is appended. If End Of File is reached: reset pointer position to fourth byte
        '''
        blocks = []
        for i in range(batch_size): # iteriert batch_size Mal, in jeder Iteration wird ein Block aus dem filebuffer extrahiert
            len_ = self.fb.read(4) # For each iteration, the NEXT 4 bytes are read from the file buffer.  Importantly, this action advances the file pointer by 4 bytes.
            if len(len_) != 4: # if fewer than 4 bytes are returned, the file pointer is reset to the 4th byte from the start (self.fb.seek(4)). This could be a way to loop back to the beginning of the file if the end is reached, although this is speculative.
                self.fb.seek(4)
                len_ = self.fb.read(4) #Importantly, this action advances the file pointer by 4 bytes.
            len_ = int.from_bytes(len_, 'little') 
            enc = self.fb.read(len_) #read actual block data. This action advances the file pointer by len_ bytes!
            block = np.load(BytesIO(enc), allow_pickle=True) # Deserialize the block (make it from bytes into a list)
            blocks.append(block)
        return np.array(blocks)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_data)#len(self.list_IDs)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self): #list_IDs_temp
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
#        X = np.empty((self.batch_size, *self.dim, self.n_channels))
#        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        index=np.arange(self.buffer.shape[0]) #index is an array containing integers from 0 to self.buffer.shape[0] - 1, where self.buffer.shape[0] would be the number of samples in the buffer.
        sample_index = np.random.choice(index, self.batch_size, replace=False) #sample_index is an array of unique random integers sampled from index, with the size of self.batch_size. These are the indices of the samples that will be included in the next batch.
        # replace=False argument ensures that once an index is selected, it won't be selected again in subsequent draws.
        next_batch = self.load_batch(self.batch_size)   # here we are loading one batch from the bytestream file (pickle)
        out_batch = self.buffer[sample_index] # here we are taking out the blocks which we randomly chose from the buffer
        self.buffer[sample_index] = next_batch # here we fill the buffer with the data we just loaded from the pickle, after we removed the data during out_batch

        #The operation essentially refreshes the in-memory buffer by swapping out the data that was just used for training or evaluation (out_batch) with new data (next_batch).
        
        if self.noise == True:
          out_batch_y=out_batch
          out_batch=random_noise(out_batch,mode='speckle')
              
        else: 
          out_batch_y=out_batch
        

        return tf.expand_dims(out_batch,axis=-1),tf.expand_dims(out_batch_y,axis=-1) 