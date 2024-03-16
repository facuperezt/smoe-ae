import os
import cv2
import pickle
import numpy as np
from skimage.util.shape import view_as_windows
from PIL import Image


# Constants
BLOCK_SIZE = 8
step_val = 8

# Paths
folder_images_path = "/home/elmoghazi/Masterarbeit/Data/pickles/test/tobepickled/"
folder_pickles_path = "/home/elmoghazi/Masterarbeit/Data/pickles/test/"



def open_and_process_image(img_path, block_size, step_val):

    img = Image.open(img_path).convert("L")  # Open image in grayscale mode
    img_array= np.array(img)
    shape_orig = (img_array.shape[0], img_array.shape[1]) 
    shape_orig_blocked = (int(img_array.shape[0]/block_size), int(img_array.shape[1]/block_size), block_size, block_size)
    #img_blocked  = preprocess_image(img=img_array, block_size=block_size, step_val=step_val)
    if step_val == block_size:
        img_blocked  = preprocess_image(img=img_array, block_size=block_size, step_val=step_val)        
    
    else:
         #shape = (img_array.shape[0], img_array.shape[1], img_blocked.shape[0]* img_blocked.shape[2], img_blocked.shape[1]*img_blocked.shape[3], img_blocked.shape[0], img_blocked.shape[1], img_blocked.shape[2], img_blocked.shape[3])
         img_blocked, overshoot_x, overshoot_y  = preprocess_image(img=img_array, block_size=block_size, step_val=step_val)
         shape_step_val = (img_blocked.shape[0]* img_blocked.shape[2], img_blocked.shape[1]*img_blocked.shape[3])
         shape_step_val_blocked = (img_blocked.shape[0], img_blocked.shape[1], img_blocked.shape[2], img_blocked.shape[3])
         img_blocked = np.reshape(img_blocked, (img_blocked.shape[0]*img_blocked.shape[1], block_size, block_size))
         return img_blocked, shape_orig, shape_orig_blocked, shape_step_val, shape_step_val_blocked, overshoot_x, overshoot_y
    
    return img_blocked, shape_orig, shape_orig_blocked

def preprocess_image(img, block_size, step_val):  
    """
    rest_x: is amount of pixels that don't get windowed in x direction when shapes misalign.
    rest_y: is amount of pixels that don't get windowed in x direction when shapes misalign.
    pad_x: amount of pixels that need to be appended in x direction to the end of an image to include all pixels when windowing.
    pad_y: amount of pixels that need to be appended in y direction to the end of an image to include all pixels when windowing.
    overshoot_x: amount of dummy pixels at the end of an image, when windowing after padding. in x direction.
    overshoot_y: amount of dummy pixels at the end of an image, when windowing after padding. in y direction.
    """ 
    img_normed = img/255

    if step_val == block_size:
        img_blocks = view_as_windows(img_normed, (block_size, block_size), step=step_val) 
        img_blocks = np.array(img_blocks)
        return img_blocks
    else:
        rest_x = (img.shape[0]-block_size)%(step_val)
        rest_y = (img.shape[1]-block_size)%(step_val)
        pad_x = block_size - rest_x
        pad_y = block_size - rest_y
        overshoot_x = step_val - rest_x
        overshoot_y = step_val -rest_y
        if pad_x == block_size:
            pad_x = 0
            overshoot_x = 0
        if pad_y == block_size:
            pad_y = 0
            overshoot_y = 0

        padded_img = np.pad(img_normed, ((0,pad_x), (0,pad_y)), mode='constant')
        img_blocks = view_as_windows(padded_img, (block_size, block_size), step=step_val)
        img_blocks = np.array(img_blocks)
        return img_blocks, overshoot_x, overshoot_y
    
    
#test = img_blocks.transpose(0,2,1,3).reshape(img_blocks.shape[0]*block_size, img_blocks.shape[1]*block_size)



# Fail because after removing pixels, img cannot be reshaped into (X,Y,block_size,block_size)
#img_blocks = img_blocks.transpose(0,2,1,3).reshape(img_blocks.shape[0]*block_size, img_blocks.shape[1]*block_size)
#img_blocks = img_blocks[:img_blocks.shape[0]-overshoot_x, :img_blocks.shape[1]-overshoot_y]
#img_blocks = img_blocks.transpose(0,2,1,3).reshape(img_blocks.shape[0]/block_size, img_blocks.shape[2]/block_size, block_size, block_size)
#img_blocks = view_as_windows(img_normed, (block_size, block_size),step=step_val) 



for filename in os.listdir(folder_images_path):
    img_path = os.path.join(folder_images_path, filename)

    if step_val == BLOCK_SIZE:
        img_blocked, shape_orig, shape_orig_blocked = open_and_process_image(img_path=img_path, block_size=BLOCK_SIZE, step_val=step_val)
        # Prepare data to pickle
        data_to_pickle = {
            "img_blocked": img_blocked, #blocked img data
            "shape_orig": shape_orig, #shape of the original img
            "shape_orig_blocked": shape_orig_blocked #shape of the original img after blocking
        }
    else:
        img_blocked, shape_orig, shape_orig_blocked, shape_step_val, shape_step_val_blocked, overshoot_x, overshoot_y = open_and_process_image(img_path=img_path, block_size=BLOCK_SIZE, step_val=step_val)
        
        data_to_pickle = {
            "img_blocked": img_blocked, #blocked img data
            "shape_orig": shape_orig, #shape of the original img
            "shape_orig_blocked": shape_orig_blocked, #shape of the original img after blocking 
            "shape_step_val": shape_step_val, #shape of the img after windowing with step_val  and reshaping to width x height
            "shape_step_val_blocked": shape_step_val_blocked, #shape of the img after windowing with step_val
            "overshoot_x": overshoot_x, #overshoot of the img in x dimension, that is the amount of dummy pixels at the end of the img, that are not part of the original img
            "overshoot_y": overshoot_y #overshoot of the img in y dimension, that is the amount of dummy pixels at the end of the img, that are not part of the original img
        }

    filename_root, _= os.path.splitext(filename)
    img_pickle_path = os.path.join(folder_pickles_path, filename_root + ".pckl")
    with open(img_pickle_path, 'wb') as f:
        pickle.dump(data_to_pickle, f)


