import gc 
import os
from os import walk
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras.layers import Input, Flatten, Lambda, Dense, Concatenate
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from skimage.metrics import structural_similarity as compare_ssim
#Scripts
import functions as fn
import classes as cl
#tf.config.run_functions_eagerly(True)


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device,True)



#Constants
BLOCK_SIZE = int(8)
NUM_KERNELS = 4 #int(4*(BLOCK_SIZE/16)*(BLOCK_SIZE/16))
STEP_VAL = int(8) #from blocked_images_bitstream_windows.py 
BATCH_SIZE = int(50)
EPOCHS = int(60)
BUFFER_SIZE = int(200)
User = "elmoghazi"



#Path
path_train_8_dg = "/home/"+User+"/Masterarbeit/facundo/Trainingdata/block_size-8_step-8_normalize-False_augmentation-True.train"
path_valid_8_dg = "/home/"+User+"/Masterarbeit/facundo/Trainingdata/block_size-8_step-8_normalize-False_augmentation-True.valid"
test_image_8_path = "/home/"+User+"/Masterarbeit/facundo/pickles/"

#Datagenerator
train_dg, valid_dg = fn.load_and_create_datagenerator(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE, path_train=path_train_8_dg, path_valid=path_valid_8_dg)


#Model
encoder_input = Input(shape=(BLOCK_SIZE,BLOCK_SIZE,1), name="encoder_input")
model_no_Smoe = fn.standard_8(x=encoder_input, latent_depth=NUM_KERNELS*6)
model_no_Smoe = Model(inputs=encoder_input, outputs=model_no_Smoe)
smoe = cl.SmoeLayer(num_kernels=NUM_KERNELS, block_size=BLOCK_SIZE)(model_no_Smoe.output)
encoder_model = keras.Model(inputs=encoder_input, outputs=smoe)

#lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5, decay_steps=100000, decay_rate=0.96)
visualize_callback = cl.VisualizeCallback(test_image_path=test_image_64_path, block_size=BLOCK_SIZE, num_kernels=NUM_KERNELS, model_callback=model_no_Smoe, step_val=STEP_VAL)
loss_history_callback = cl.LossHistoryCallback()

encoder_model.summary()
encoder_model.compile(optimizer=keras.optimizers.Adam(
    learning_rate=0.00005),
    loss = tf.keras.losses.MeanSquaredError()
    )

encoder_model.fit(train_dg, epochs=EPOCHS, validation_data=valid_dg,
          callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
            visualize_callback,
            loss_history_callback
          ]
)

encoder_model.save(save_model_path)

