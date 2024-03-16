import numpy as np
from os import walk
from skimage.util.shape import view_as_blocks
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
#from random import seed
#from random import randint
from random import uniform
import pickle
import cv2
from io import BytesIO
#from scipy.ndimage import gaussian_filter
from tqdm import tqdm






def save_py(path,data_name, serialized_file):
    """
    path: Path to location, where pickle should be stored
    data_name: name of the data 
    img_blocked: the blocked images
    """
    cp = {'block':serialized_file}
        #'mask':mask,
        #'w':w_e_op,
        #'A_init':sigma_x}
    sav = open(path+'blocked/'+str(data_name)+'.pckl', 'wb')#'just-center-anzahl'+str(anzahl)+'-sigma_'+str(sigma_x).replace('.','-')+'_gaussblur_3-3-withoutamax'+'_params.pckl'
    pickle.dump(cp, sav, protocol=pickle.HIGHEST_PROTOCOL)
    sav.close()

def normalize(value):
    """
    Function removes the mean from an array and then normalizes it by dividing by 255
    """
    summe = np.sum(value)
    mean = (summe / (value.shape[0] * value.shape[1]))
    pixel_meanfree = value - mean
    #max = np.amax(np.absolute(pixel_meanfree))
    value_norm = pixel_meanfree/255

    return value_norm#, max, mean

def denormalize(mean,max,value_norm):
    """
    Takes a normalizes array and denormalizes it by multiplying by 255 and adding the mean. Mean is a function Parameter.
    """
    temp_value = value_norm*max
    value = temp_value + mean
    return value

#path=os.getcwd()

def resize_img(filenames,data_file_path,pic_num,normalize_bool):
    """
    Filenames: Array that holds the name of all pictures.
    myPath: Path on Computer in which the Files are stored.
    pic_num: index of pic which we want to load
    normalize_bool: bool, that indicate wether we want to normalize_bool the picture.

    This function returns the resized 512x512 image and the name of the image that we specify using the pic_num
    """
    filename = filenames[pic_num]
    path_img=data_file_path+filename
    img_rgb = cv2.imread(path_img, 1) #1 = rgb
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #convert to grayscale

    # Checks the aspect ratio of the image and resizes it such that the smaller dimension becomes 512 pixels while maintaining the aspect ratio.
    if img_gray.shape[1]<=img_gray.shape[0]:
        ratio = 512.0 / img_gray.shape[1]
        dim = (512, int(img_gray.shape[0] * ratio))
    else:
        ratio = 512.0 / img_gray.shape[0]
        dim = (int(img_gray.shape[1] * ratio),512)
    
    
    img_resize = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
    # now we have a reshaped image: smaller dimension is 512 pixels, bigger dimension is ratio * itself.
    img_temp = img_resize[0:512,0:512] #Takes a 512x512 pixel region from the top-left corner of the resized image.
    if normalize_bool==True:
        image_0 = normalize(img_temp) #normalize_bool and remove mean
    else:
        image_0=img_temp/255 #normalize_bool but don't subtract mean
    return image_0, filename


def make_dataset(block_size,type,normalize_bool,step_val,augmentation_bool=False,tresh_bool=False,tresh_val=0):
    """
    tresh_bool / tresh_val: tres potentially stands for threshold. Threshold in regards to std deviation. If std deviation of one block of data is too small, it is not being used.
    block_size: size of the blocks, in which we chop each image
    type: "train" or "valid"
    normalize_bool: bool, if true remove mean and normalize_bool by 255, if false normalize_bool by 255
    step_val: potentially the step of a sliding window

    Function creates a sequential binary file out of all of the images in either train or valid directory.  Before they are stored the images are chopped into blocks and can be augmented.
    Bytes 0-4: Number of Blocks, Bytes 4-8: Size of Block1, Bytes A-A+4 Size of Block2...
    
    """

    # Naming of the file and storing it in data_file_name
    # set path where images are stored for train and valid
    if type=='train':
        if tresh_bool:
            data_file_name='block_size-'+str(block_size)+'_tres-'+str(tresh_val)+'_step-'+str(step_val)+'_normalize-'+str(normalize_bool)+ '_augmentation-'+str(augmentation_bool)+'.train'
        else:
            data_file_name='block_size-' + str(block_size) + '_step-' + str(step_val) + '_normalize-' + str(
                normalize_bool) + '_augmentation-'+str(augmentation_bool)+ '.train'
        data_file_path = '/home/elmoghazi/Masterarbeit/Data/train/'
    elif type=='valid':
        if tresh_bool:
            data_file_name='block_size-'+str(block_size)+'_tres-'+str(tresh_val)+'_step-'+str(step_val)+'_normalize-'+str(normalize_bool)+ '_augmentation-'+str(augmentation_bool)+'.valid'
        else:
            data_file_name='block_size-' + str(block_size) + '_step-' + str(step_val) + '_normalize-' + str(
                normalize_bool) + '_augmentation-'+str(augmentation_bool)+ '.valid'
        data_file_path = '/home/elmoghazi/Masterarbeit/Data/valid/'


    dir_data_generator = '/home/elmoghazi/Masterarbeit/Data/pickles/dataset/Datagenerator/'
    full_path_data_generator = dir_data_generator + data_file_name

    data_size_placeholder=2000000000 #2*10e9
    num_valid_blocks=0

    # store all filenames in data_file_path in list
    image_names = []
    for (dirpath, dirnames, filenames) in walk(data_file_path):
        image_names.extend(filenames)
        break
    

    with open(full_path_data_generator, 'wb') as fd:
        fd.write(data_size_placeholder.to_bytes(4, 'little')) #first four bytes of file: say how big the file is BUT this is a placeholder. Will later (at the end of the function) be overwritten with number of num_valid_blocks

        #iterates through each filename stored in list image_names. Tqdm = arabic taqqadum for progress bar
        # loads each image as a 512x512 array
        for i in tqdm(range(len(image_names))):
            img,filename=resize_img(image_names,data_file_path,i,normalize_bool)
            img_blocked = view_as_windows(img, (block_size, block_size),step=step_val) #view_as_windows is presumably from skimage.util. when step_val is set to 16, then we get a (32,32,16,16) array. When step_val is lower bigger array is returned, as it repeats
            img_blocked = img_blocked.reshape((-1, block_size, block_size))
            num_blocks = img_blocked.shape[0]

            # if tresh_bool exists: blocks with std deviation that is too small are removed 
            if tresh_bool:
                blocks_over_thresh = []
                for a in range(num_blocks):
                    std_dev=(img_blocked[a].std())*255 # standard deviation of this block. It quantifies how spread out the data points in the dataset are from the mean (average) value
                    if std_dev>tresh_val:
                        blocks_over_thresh.append(img_blocked[a])
                num_blocks = np.array(blocks_over_thresh).shape[0]
                img_blocked=np.array(blocks_over_thresh)

            # augments the data in both axis (1,2) and appends them to the original array. Augmented Data shouldn't replace the original data, that's why we append it.
            if augmentation_bool and img_blocked.tolist():
                img_aug_01=np.flip(img_blocked,axis=1) # flip each block horizonally
                img_temp=np.concatenate((img_blocked,img_aug_01),axis=0)
                img_aug_02=np.flip(img_temp,axis=2) # flip each block vertically
                img_blocked=np.concatenate((img_temp,img_aug_02),axis=0)
                num_blocks = img_blocked.shape[0]

            # this looks like old code, num_valid_blocks is set to 0 above therefore num_valid_blocks = num_blocks and blocks_per_dim is unused.
            num_valid_blocks = num_valid_blocks + num_blocks
            blocks_per_dim = int(np.sqrt(num_blocks))

            #num_blocks_over_thres
            #print(num_blocks)
            #data_name=filename+'.test'
            #save_py(data_file_path.format(block=img_blocked),filename.replace('.png',''))



            # This loop is responsible for writing each block of the image to a file in a serialized form
            for block in range(num_blocks):
                buffer = BytesIO() #creates an in-memory binary stream. Think of it as a file-like object that resides in memory instead of being written to disk. 
                np.save(buffer, img_blocked[block]) #uses NumPy's save function to serialize (convert to a byte stream) each block from img_blocked and saves it into the buffer object
                encoded = buffer.getvalue()
                fd.write(len(encoded).to_bytes(4, 'little')) #writes the length of the serialized block to the file, using 4 bytes
                fd.write(encoded) #writes the serialized block that was in memory into the fd file
            fig, (ax1, ax2) = plt.subplots(1, 2)
            #flatten_view=img_blocked.reshape(32,32,-1)
            #flatten_view=img_blocked.reshape(32,32,16,16).transpose(0,2,1,3).reshape(512,512)
            #ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
            #ax2.imshow(flatten_view, cmap='gray', vmin=0, vmax=1)  #
            #plt.show()

        #At the very end, the code performs a fd.seek(0) to go back to the start of the file and overwrites the initial 4 bytes with the total number of valid blocks 
        fd.seek(0)
        print('valid block:' + str(num_valid_blocks))
        fd.write(num_valid_blocks.to_bytes(4, 'little'))
        #fd.close() no need to clsoe file since I'm using with statement

        #save_py(path="/home/tarek/Masterarbeit/Data/pickles/dataset/Datagenerator/",data_name=data_file_name, serialized_file=fd)

        #print(img_blocked)

block_size = 8
step_val = 8
make_dataset(block_size=block_size, type='train', normalize_bool=False, step_val=step_val, augmentation_bool=True)
make_dataset(block_size=block_size, type='valid', normalize_bool=False, step_val=step_val, augmentation_bool=True)
