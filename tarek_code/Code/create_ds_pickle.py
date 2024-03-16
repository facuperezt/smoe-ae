import os
import cv2
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

"""
np.sum(avg_shannon_per_img) / avg_shannon_per_img.shape[0]
6.3028417935759675
np.max(avg_shannon_per_img)
7.607236857307321
np.min(avg_shannon_per_img)
2.925444570853942
np.median(avg_shannon_per_img)
6.38381741523828

"""

# Constants
BLOCK_SIZE = 128
IMG_DIM  = 512




def load_and_process_images(folder_path, img_dim, block_size):
    blocks = []
    #avg_shannon_per_img = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("L")  # Open image in grayscale mode
        img_array = np.array(img)
        img_blocks= preprocess_image(img_array, img_dim = img_dim, block_size=block_size)
        #avg_shannon_per_img += [shannon]
        blocks.append(img_blocks)
    #avg_shannon_per_img= np.array(avg_shannon_per_img)
    blocks_array = np.array(blocks)
    return np.reshape(blocks_array,(blocks_array.shape[0]*blocks_array.shape[1],block_size,block_size))

def preprocess_image(img, img_dim, block_size):   

    if img.shape[1]<=img.shape[0]:
        ratio = img_dim / img.shape[1]
        dim = (img_dim, int(img.shape[0] * ratio))
    else:
        ratio = img_dim / img.shape[0]
        dim = (int(img.shape[1] * ratio), img_dim)

    img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img_resize = img_resize[0:img_dim,0:img_dim]
    img_normed = img_resize /255

    # Create 16x16 blocks
    img_blocks = [img_normed[i:i + block_size, j:j + block_size] for i in range(0, img_dim, block_size) for j in range(0, img_dim, block_size)]
    img_blocks = np.array(img_blocks)
    #shannon = []
    #for i in range(img_blocks.shape[0]):
    #    shannon += [calculate_complexity(img_blocks[i,:])]

    return img_blocks#, sum(shannon)/len(shannon)



def calculate_complexity(block):
    fft_block = np.fft.fft2(block)
    magnitude_spectrum = np.abs(fft_block)
    # Calculate the total spectral energy
    total_energy = np.sum(magnitude_spectrum[1:,1:]**2)
    energy_pixel = total_energy / ((BLOCK_SIZE*BLOCK_SIZE)-1)
    entropy_value = shannon_entropy(block)
    """
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.tick_params(axis='both', labelsize=5)
    ax1.imshow(block, cmap='gray')
    ax1.set_title(f'Shannon Entropy:: {entropy_value:.2f}') 
    ax2.tick_params(axis='both', labelsize=5)
    ax2.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    #ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.set_title(f'Mean Energy per Pixel:: {energy_pixel:.2f}') 
    #plt.show(block=True)
    """
    return entropy_value

path_train = "/home/tarek/Masterarbeit/Data/train"
path_valid = "/home/tarek/Masterarbeit/Data/valid"

processed_blocks_train = load_and_process_images(folder_path=path_train, img_dim=IMG_DIM, block_size=BLOCK_SIZE)
processed_blocks_valid = load_and_process_images(folder_path=path_valid, img_dim=IMG_DIM, block_size=BLOCK_SIZE)

# Split Train set in two arrays. Too large arrays cause Problems when pickling
#size = processed_blocks_train.shape[0] // 2
#processed_blocks_train_first_half = processed_blocks_train[:size,:,:]
#processed_blocks_train_second_half = processed_blocks_train[size:,:,:]


# Save processed blocks to a file using pickling
'''
with open('/home/tarek/Masterarbeit/Data/pickles/dataset/processed_blocks_train_first_half.pkl', 'wb') as f:
    pickle.dump(processed_blocks_train_first_half, f)

with open('/home/tarek/Masterarbeit/Data/pickles/dataset/processed_blocks_train_second_half.pkl', 'wb') as f:
    pickle.dump(processed_blocks_train_second_half, f)
'''

with open('/home/tarek/Masterarbeit/Data/pickles/dataset/processed_blocks_train.pkl', 'wb') as f:
    pickle.dump(processed_blocks_train, f)
#with open('/home/tarek/Masterarbeit/Data/pickles/dataset/processed_blocks_valid.pkl', 'wb') as f:
#    pickle.dump(processed_blocks_valid, f)

