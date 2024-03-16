import gc
import pickle
from os import walk
import tensorflow as tf
#import tensorflow_compression as tfc
from tensorflow import keras
import numpy as np
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from math import sqrt
#from keras import mixed_precision
#Scripts
from . import functions as fn
import functools
#tf.compat.v1.disable_eager_execution()
#tf.config.run_functions_eagerly(True)
# mixed_precision.set_global_policy('mixed_float16')



# start classes
class VisualizeCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_image_path, block_size, num_kernels, model_callback, step_val):
        super(VisualizeCallback, self).__init__()
        self.test_image_path = test_image_path
        self.block_size = block_size
        self.num_kernels = num_kernels
        self.model_callback = model_callback
        self.step_val = step_val

    def on_epoch_end(self, epoch, logs=None): 
        block_size = self.block_size
        num_kernels = self.num_kernels
        #img_dim_blocked = int(sqrt(img_dim**2/block_size/block_size))
        model_callback = self.model_callback

        if epoch !=300:

            g = []
            for (_, _, filenames) in walk(self.test_image_path):
                g.extend(filenames)
                break

            
            for i in range(4):
                filename = g[i]
                data = pickle.load(open(self.test_image_path + filename, "rb"))
                if self.step_val == block_size:
                    img_blocked = data["img_blocked"]
                    img_blocked = np.reshape(img_blocked, (-1, block_size, block_size))
                    shape_orig = data["shape_orig"]
                    shape_orig_blocked = data["shape_orig_blocked"]
                    img_orig = np.reshape(img_blocked,(shape_orig_blocked[0], shape_orig_blocked[1], block_size, block_size)).transpose(
                    0, 2, 1, 3).reshape(shape_orig[0], shape_orig[1])
                    features = model_callback.predict(img_blocked)
                    img_reconst = fn.Smoe_reconst_og(num_kernels=num_kernels,block_size=block_size, center=features).reshape(shape_orig_blocked).transpose(0,2,1,3).reshape(shape_orig)
 
                    psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
                    ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)

                    fig, ((ax1, ax2)) = plt.subplots(1, 2)
                    ax1.tick_params(axis='both', labelsize=5)
                    ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
                    ax1.set_title('Original')
                    ax2.tick_params(axis='both', labelsize=5)
                    ax2.imshow(img_reconst, cmap='gray',
                            vmin=0, vmax=1)
                    ax2.set_title(
                    'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
                    round(ssim_reconst, 2)))


			
                    plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/Trainingpics/epoch_'+str(epoch)+"_counter_"+str(i)+".png") # You need to change '/path/to/your/directory/your_image_name.png' to your actual path and file name
                    plt.close(fig) # Close the figure to free up memory

                   # plt.show(block=False)
                
                else:
                    img_blocked = data["img_blocked"]
                    shape_orig = data["shape_orig"]
                    shape_orig_blocked = data["shape_orig_blocked"]
                    shape_step_val = data["shape_step_val"]
                    shape_step_val_blocked = data["shape_step_val_blocked"]
                    overshoot_x = data["overshoot_x"]
                    overshoot_y = data["overshoot_y"]
                    img_orig = fn.average_overlap(img_blocked.reshape(shape_step_val_blocked[0], shape_step_val_blocked[1], block_size, block_size), block_size=block_size, step_val=self.step_val, shape_orig=shape_orig, b_average=False)
                    features = model_callback.predict(img_blocked)
                    img_reconst_overlap = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=num_kernels, block_size=block_size, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=shape_step_val_blocked[1], step_val=self.step_val, b_average=True)
                    img_reconst = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=num_kernels, block_size=block_size, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=shape_step_val_blocked[1], step_val=self.step_val, b_average=False)
                    psnr_reconst_overlap = fn.psnr(img_A=img_orig, img_B=img_reconst_overlap)
                    psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
                    ssim_reconst_overlap = compare_ssim(img_orig, img_reconst_overlap, data_range=1.)
                    ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)
                
                    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
                    ax1.tick_params(axis='both', labelsize=5)
                    ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
                    ax1.set_title('Original',fontsize=8)
                    ax2.tick_params(axis='both', labelsize=5)
                    ax2.imshow(img_reconst, cmap='gray',
                            vmin=0, vmax=1)
                    ax2.set_title(
                    'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
                    round(ssim_reconst, 2)), fontsize=8)

                    ax3.tick_params(axis='both', labelsize=5)
                    ax3.imshow(img_reconst_overlap, cmap='gray', vmin=0, vmax=1)
                    ax3.set_title(
                                    'psnr_smooth: ' + str(round(psnr_reconst_overlap, 2)) + ', ssim_smooth: ' + str(
                    round(ssim_reconst_overlap, 2)),fontsize=8)


                    plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/Trainingpics/epoch_'+str(epoch)+"_counter_"+str(i)+".png") # You need to change '/path/to/your/directory/your_image_name.png' to your actual path and file name
                    plt.close(fig) # Close the figure to free up memory

                    #plt.show(block=False)


    def on_batch_end(self, batch, logs=None): 
        if batch == 500 or batch == 2500 or batch==5000 or batch==10000 or batch==15000 or batch==20000 or batch==30000 or batch==40000 or batch == 50000 or batch == 60000 or batch == 70000 or batch== 90000 or batch == 100000 or batch == 200000 or batch == 500000 or batch == 700000:
            block_size = self.block_size
            num_kernels = self.num_kernels
            #img_dim_blocked = int(sqrt(img_dim**2/block_size/block_size))
            model_callback = self.model_callback

            g = []
            for (_, _, filenames) in walk(self.test_image_path):
                g.extend(filenames)
                break

            
            for i in range(4):
                filename = g[i]
                data = pickle.load(open(self.test_image_path + filename, "rb"))
                if self.step_val == block_size:
                    img_blocked = data["img_blocked"]
                    img_blocked = np.reshape(img_blocked, (-1, block_size, block_size))
                    shape_orig = data["shape_orig"]
                    shape_orig_blocked = data["shape_orig_blocked"]
                    img_orig = np.reshape(img_blocked,(shape_orig_blocked[0], shape_orig_blocked[1], block_size, block_size)).transpose(
                    0, 2, 1, 3).reshape(shape_orig[0], shape_orig[1])
                    features = model_callback.predict(img_blocked)
                    #img_reconst = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=num_kernels, block_size=block_size, img_dim_blocked_x=shape_orig_blocked[0], img_dim_blocked_y=shape_orig_blocked[1], step_val=self.step_val)
                    img_reconst = fn.Smoe_reconst_og(num_kernels=num_kernels,block_size=block_size, center=features).reshape(shape_orig_blocked).transpose(0,2,1,3).reshape(shape_orig)

                    psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
                    ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)

                    fig, ((ax1, ax2)) = plt.subplots(1, 2)
                    ax1.tick_params(axis='both', labelsize=5)
                    ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
                    ax1.set_title('Original')
                    ax2.tick_params(axis='both', labelsize=5)
                    ax2.imshow(img_reconst, cmap='gray',
                            vmin=0, vmax=1)
                    ax2.set_title(
                    'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
                    round(ssim_reconst, 2)))


                    plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/Trainingpics/batch_'+str(batch)+"_counter_"+str(i)+".png") # You need to change '/path/to/your/directory/your_image_name.png' to your actual path and file name
                    plt.close(fig) # Close the figure to free up memory
                    #plt.show(block=False)
                
                else:
                    img_blocked = data["img_blocked"]
                    shape_orig = data["shape_orig"]
                    shape_orig_blocked = data["shape_orig_blocked"]
                    shape_step_val = data["shape_step_val"]
                    shape_step_val_blocked = data["shape_step_val_blocked"]
                    overshoot_x = data["overshoot_x"]
                    overshoot_y = data["overshoot_y"]
                    img_orig = fn.average_overlap(img_blocked.reshape(shape_step_val_blocked[0], shape_step_val_blocked[1], block_size, block_size), block_size=block_size, step_val=self.step_val, shape_orig=shape_orig, b_average=False)
                    features = model_callback.predict(img_blocked)
                    img_reconst_overlap = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=num_kernels, block_size=block_size, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=shape_step_val_blocked[1], step_val=self.step_val, b_average=True)
                    img_reconst = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=num_kernels, block_size=block_size, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=shape_step_val_blocked[1], step_val=self.step_val, b_average=False)
                    psnr_reconst_overlap = fn.psnr(img_A=img_orig, img_B=img_reconst_overlap)
                    psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
                    ssim_reconst_overlap = compare_ssim(img_orig, img_reconst_overlap, data_range=1.)
                    ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)
                
                    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
                    ax1.tick_params(axis='both', labelsize=5)
                    ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
                    ax1.set_title('Original',fontsize=6)
                    ax2.tick_params(axis='both', labelsize=5)
                    ax2.imshow(img_reconst, cmap='gray',
                            vmin=0, vmax=1)
                    ax2.set_title(
                    'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
                    round(ssim_reconst, 2)),fontsize=6)

                    ax3.tick_params(axis='both', labelsize=5)
                    ax3.imshow(img_reconst_overlap, cmap='gray', vmin=0, vmax=1)
                    ax3.set_title(
                                    'psnr_smooth: ' + str(round(psnr_reconst_overlap, 2)) + ', ssim_smooth: ' + str(
                    round(ssim_reconst_overlap, 2)),fontsize=6)

                    plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/Trainingpics/batch_'+str(batch)+"_counter_"+str(i)+".png") # You need to change '/path/to/your/directory/your_image_name.png' to your actual path and file name
                    plt.close(fig) # Close the figure to free up memory
                    #plt.show(block=False)


class SmoeLayer(keras.layers.Layer):
    def __init__(self, num_kernels=4, block_size=16, step_val=None):
        super(SmoeLayer, self).__init__()
        self.num_kernels = num_kernels
        self.block_size = block_size
        self.step_val = step_val if step_val is not None else block_size


    def call(self, inputs):
        block_size = self.block_size
        num_kernels = self.num_kernels

        x = np.linspace(0, 1, block_size).astype(dtype=np.float32)
        y = np.linspace(0, 1, block_size).astype(dtype=np.float32)
        domain_init = np.array(np.meshgrid(x, y)).T
        domain_init = domain_init.reshape([block_size ** 2, 2])  
        domain_init = tf.cast(domain_init, dtype=tf.float32)
        center_x = inputs[:, 0:num_kernels]  
        center_y = inputs[:, num_kernels:2 * num_kernels]  
        nue_e = inputs[:, 2 * num_kernels:3 * num_kernels]  

        chol_diag = inputs[:, 3*num_kernels:5*num_kernels]
        chol_diag = tf.reshape(chol_diag, [-1, num_kernels, 2])
        chol_offdiag = inputs[:, 5*num_kernels:]
        chol_offdiag = tf.reshape(chol_offdiag, [-1, num_kernels, 1])
        zeros = tf.zeros_like(chol_offdiag)
        upper = tf.concat([chol_diag[:, :, :1], zeros], axis=-1)
        lower = tf.concat([chol_offdiag, chol_diag[:, :, 1:]], axis=-1)
        chol = tf.concat([upper, lower], axis=-1)
        chol = tf.reshape(chol, [-1, num_kernels, 2, 2])
        A_NN = chol
        #del chol, chol_diag, chol_offdiag, zeros, upper, lower
        #gc.collect()

        shape_x = tf.shape(inputs)
        reshape_x = tf.reshape(center_x, (shape_x[0], num_kernels, 1)) #(batch_size,num_kernels,1)
        reshape_y = tf.reshape(center_y, (shape_x[0], num_kernels, 1)) #(batch_size,num_kernels,1)
        centers = tf.reshape(
            tf.concat([reshape_x, reshape_y], axis=2), (shape_x[0], num_kernels, 2))  # centers (x&y): (batch_size,num_kernels,2)
        
        
        musX = tf.expand_dims(centers, axis=2) #(batch_size,num_kernels,1,2)
        domain_exp = tf.tile(tf.expand_dims(tf.expand_dims(domain_init, axis=0), axis=0),
                             (tf.shape(musX)[0], tf.shape(musX)[1], 1, 1))  # tf.tile(domain_init(1,1,block_size**2,2), (batch_size, num_kernels,1,1)
        x_sub_mu = tf.expand_dims(tf.cast(domain_exp, dtype=tf.float32) - tf.cast(musX, dtype=tf.float32),
                                  axis=-1)
        x_sub_mu = tf.cast(x_sub_mu, dtype=tf.float32)
        A_NN = tf.cast(A_NN, dtype=tf.float32)

        n_exp = tf.exp(
            tf.negative(0.5 * tf.einsum('abcli,ablm,abnm,abcnj->abc', tf.cast(x_sub_mu, dtype=tf.float32), tf.cast(A_NN, dtype=tf.float32), tf.cast(A_NN, dtype=tf.float32), tf.cast(x_sub_mu, dtype=tf.float32))))
        n_w_norm = tf.reduce_sum(n_exp, axis=1, keepdims=True) #Sum over all Kernels
        n_w_norm = tf.maximum(tf.cast(10e-10, dtype=tf.float32), n_w_norm) # numeric stability, perhaps set to 10e-10

        w_e_op = tf.divide(n_exp, n_w_norm, name="skdjfbk")
        # nue_e=tf.squeeze(OLS(w_e_op,block))
        res = tf.reduce_sum(
            w_e_op * tf.expand_dims(tf.cast(nue_e, dtype=tf.float32), axis=-1), axis=1)
        res = tf.minimum(tf.maximum(res, 0), 1)
        res = tf.reshape(res, (-1, block_size, block_size))

        return tf.expand_dims(res,-1) # shape [none, 16, 16]
        #return res

        
          
class LossHistoryCallback(tf.keras.callbacks.Callback):   
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('MSE'))

        # Plot the loss curve over epochs
        plt.plot(self.losses, label='Training Loss')
        plt.title('Mean Squared Error Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/MSE/epoch_'+str(epoch)+'.png') 
        plt.close() 
          


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        result_array = np.array(self.losses) * 255*255

        # Plot the loss curve over epochs
        plt.plot(result_array, label='Training Loss')
        plt.title('Mean Squared Error Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig('/home/elmoghazi/Masterarbeit/Data/figures/MSE/epoch_'+str(epoch)+'.png') 
        plt.close() 


class LaplaceSmoeLayer(keras.layers.Layer):
    def __init__(self, num_kernels=4, block_size=16, step_val=None):
        super(LaplaceSmoeLayer, self).__init__()
        self.num_kernels = num_kernels
        self.block_size = block_size
        self.step_val = step_val if step_val is not None else block_size
        


    def __call__(self, inputs):
        block_size = self.block_size
        num_kernels = self.num_kernels

        x = tf.linspace(0.0, 1.0, block_size)
        y = tf.linspace(0.0, 1.0, block_size)
        x_mesh, y_mesh = tf.meshgrid(x, y)
        domain_init = tf.stack([tf.transpose(x_mesh), tf.transpose(y_mesh)], axis=-1)
        domain_init = tf.reshape(domain_init, [block_size**2, 2])

        center_x = inputs[:, 0:num_kernels]  
        center_y = inputs[:, num_kernels:2 * num_kernels]  
        nue_e = inputs[:, 2 * num_kernels:3 * num_kernels]  

        # output Chol
        chol_diag = inputs[:, 3*num_kernels:5*num_kernels]
        chol_diag = tf.reshape(chol_diag, [-1, num_kernels, 2])
        chol_offdiag = inputs[:, 5*num_kernels:]
        chol_offdiag = tf.reshape(chol_offdiag, [-1, num_kernels, 1])
        zeros = tf.zeros_like(chol_offdiag)
        upper = tf.concat([chol_diag[:, :, :1], zeros], axis=-1)
        lower = tf.concat([chol_offdiag, chol_diag[:, :, 1:]], axis=-1)
        chol = tf.concat([upper, lower], axis=-1)
        chol = tf.reshape(chol, [-1, num_kernels, 2, 2])
        A_NN = chol
        del chol, chol_diag, chol_offdiag, zeros, upper, lower
        gc.collect()
        
        shape_x = tf.shape(inputs)
        reshape_x = tf.reshape(center_x, (shape_x[0], num_kernels, 1)) #(batch_size,num_kernels,1)
        reshape_y = tf.reshape(center_y, (shape_x[0], num_kernels, 1)) #(batch_size,num_kernels,1)
        centers = tf.reshape(
            tf.concat([reshape_x, reshape_y], axis=2), (shape_x[0], num_kernels, 2))  # centers (x&y): (batch_size,num_kernels,2)

        musX = tf.expand_dims(centers, axis=2) #(batch_size,num_kernels,1,2)
        domain_exp = tf.tile(tf.expand_dims(tf.expand_dims(domain_init, axis=0), axis=0),
                             (tf.shape(musX)[0], tf.shape(musX)[1], 1, 1))  # tf.tile(domain_init(1,1,block_size**2,2), (batch_size, num_kernels,1,1)
        x_sub_mu = tf.expand_dims(tf.cast(domain_exp, dtype=tf.float32) - tf.cast(musX, dtype=tf.float32),
                                  axis=-1)
        x_sub_mu = tf.cast(x_sub_mu, dtype=tf.float32)
        A_NN = tf.cast(A_NN, dtype=tf.float32)

        n_exp = tf.exp(
            tf.negative(0.5 * tf.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu,A_NN,A_NN,x_sub_mu)))  # n_exp:batch_size,num_kernels,block_size**2
        
        # Laplace Part
        subtraction = (1 + tf.negative(0.5 * tf.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu,A_NN,A_NN,x_sub_mu)))
        product = subtraction * n_exp





        n_w_norm = tf.reduce_sum(product, axis=1, keepdims=True) #Sum over all Kernels
        n_w_norm = tf.maximum(10e-5, n_w_norm) # numeric stability, perhaps set to 10e-10

        w_e_op = tf.divide(product, n_w_norm, name="skdjfbk")
        # nue_e=tf.squeeze(OLS(w_e_op,block))
        res = tf.reduce_sum(
            w_e_op * tf.expand_dims(nue_e, axis=-1), axis=1)
        res = tf.minimum(tf.maximum(res, 0), 1)
        res = tf.reshape(res, (-1, block_size, block_size))

        return tf.expand_dims(res,-1) # shape [none, 16, 16]
        #return res


def calcsig(A_NN):
    #sig = tf.concat([sig, A_NN[i,j] * tf.transpose(A_NN[i,j])],0) 

    sig = tf.map_fn(lambda x: tf.map_fn(compute_product, x), A_NN)
    sig = tf.reshape(sig, tf.shape(A_NN))
    return sig


def compute_product(matrix):
    return tf.matmul(matrix, tf.transpose(matrix))



def call_kl(sig,centers,nue,A_NN):

    kl = tf.zeros((tf.shape(sig)[0],2017))
    shape = tf.shape(sig)

    for h in range(shape[0]):
        counter = 0
        for i in range(shape[1]):
            if nue[h,i] == 0:
                continue
            for j in range(i+1, sig.shape[1]):
                counter = counter+1
                kl[h,counter] = kl_mvn_tf(m0=centers[h,i,:],m1=centers[h,j,:], S0=sig[h,i,:], S1=sig[h,j,:])
                if (kl[h,counter] <= 0.01):
                    #print(h,i,j)
                    nue[h,i] = 0.7*(nue[h,i]+nue[h,j])
                    nue[h,j] = 0
                    #A_NN[h,i] = nue[h,j]*A_NN[h,j]+nue[h,i]*A_NN[h,i]
                    #centers[h,i] = nue[h,i]*centers[h,i]+nue[h,j]*centers[h,j]
                    #nue[h,i] = 0.7*(nue[h,i]+nue[h,j])
                    #nue[h,j] = 0
    return  A_NN, nue, centers


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.


    From wikipedia
    KL( (m0, S0) || (m1, S1))
            = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                    (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = tf.shape(m0)[0]
    iS1 = tf.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = tf.linalg.trace(tf.matmul(iS1,S0))
    det_term  = tf.math.log(tf.linalg.det(S1)/tf.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = tf.matmul(tf.matmul(tf.transpose(diff),tf.linalg.inv(S1)),diff) #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 



def calcsig_tf(A_NN):
    # Transpose A_NN to align the dimensions properly
    A_NN_transposed = tf.transpose(A_NN, perm=[0, 1, 3, 2])

    # Compute the element-wise multiplication with the transposed matrix
    sig = tf.matmul(A_NN, A_NN_transposed)

    return sig


def call_kl_tf(sig, centers, nue, A_NN):
    batch_size = tf.shape(sig)[0]
    num_kernels = tf.shape(sig)[1]

    # Initialize a mask tensor with True values
    for h in range(batch_size):
        for i in range(num_kernels):
            if nue[h,i] == 0:
                continue
        # Calculate KL divergence only if nue is not 0 and j > i
            for j in range(i + 1, num_kernels):
                kl_value = kl_mvn_tf(
                    m0=centers[h, i],
                    m1=centers[h, j],
                    S0=sig[h, i],
                    S1=sig[h, j]
                )
                if kl_value <= 0.1:
                    mask = tf.zeros_like(nue, dtype=tf.bool)
                    mask = tf.tensor_scatter_nd_update(mask, [[h, i]], [True])
                    nue = tf.where(mask, nue[h,i] + nue[h,j], nue)
                    mask = tf.zeros_like(nue, dtype=tf.bool)
                    mask = tf.tensor_scatter_nd_update(mask, [[h, j]], [True])
                    nue = tf.where(mask, 0.0, nue)
    return nue

def kl_mvn_tf(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.

    From Wikipedia:
    KL( (m0, S0) || (m1, S1)) = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                                       (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # Get the dimensionality of the multivariate Gaussian distribution
    N = tf.shape(m0)[0]
    N = tf.cast(N, dtype=tf.float32)


    # Compute the inverse of S1 and the difference between means
    iS1 = tf.linalg.inv(S1)
    diff = m1 - m0
    diff = tf.expand_dims(diff, axis=1)  # Reshape diff to have shape [2, 1] for proper matrix multiplication


    # Compute the three terms of the KL divergence
    tr_term = tf.linalg.trace(tf.matmul(iS1, S0))
    det_term = tf.math.log(tf.linalg.det(S1) / tf.linalg.det(S0))
    quad_term = tf.matmul(tf.matmul(tf.transpose(diff), tf.linalg.inv(S1)), diff)

    # Compute the KL divergence
    kl_divergence = 0.5 * (tr_term + det_term + quad_term - N)

    return kl_divergence
