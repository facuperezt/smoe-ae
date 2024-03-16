#import gc 
import pickle
from math import sqrt
#from os import walk
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras.layers import Dense, Conv2D
from keras.layers import Input, UpSampling2D, Concatenate, Conv2D, Activation, Dense
from skimage.metrics import structural_similarity as compare_ssim
#import tensorflow_compression as tfc
import functools
#from keras import mixed_precision
#Scripts
from . import data_generator_shuffle as dg
from . import conv as convo
from . import gdn as gdn


def psnr(img_A, img_B):
    mse = np.sum((255*img_A.astype("float") - 255*img_B.astype("float")) ** 2)
    mse /= float(img_A.shape[0] * img_A.shape[1])
    psnr=10*np.log10(((255**2)/mse))
    return psnr

def standard_8(x, latent_depth, n_layers = 100):
    conv_down_two = functools.partial(convo.SignalConv2D, corr=True, strides_down=2,
                    padding="same_zeros", use_bias=True)

    #x=layers.Conv2D(16, (3,3), padding='same',activation=gdn.GDN(name="gdn_1"))(x)
    #x=layers.Conv2D(32, (3,3), padding='same',activation=gdn.GDN(name="gdn_2"))(x)
    # x = conv_down_two(16, (3, 3), name="layer_1")(x)
    # x = conv_down_two(32, (3, 3), name="layer_2")(x)
    
    i = 0
    def update_i_return(n_layers):
        nonlocal i
        if i < n_layers:
            i += 1
            return False
        i+=1
        return True

    x = conv_down_two(16, (3, 3), name="layer_1", activation=gdn.GDN(name="gdn_1"))(x)
    if update_i_return(n_layers): return x

    x = conv_down_two(32, (3, 3), name="layer_2", activation=gdn.GDN(name="gdn_2"))(x)
    if update_i_return(n_layers): return x

    x=layers.Conv2D(64, (3,3), padding='same',activation=gdn.GDN(name="gdn_3"))(x)
    if update_i_return(n_layers): return x

    x=layers.Conv2D(128, (3,3), padding='same',activation=gdn.GDN(name="gdn_4"))(x)
    if update_i_return(n_layers): return x

    x=layers.Conv2D(256, (3,3), padding='same',activation=gdn.GDN(name="gdn_5"))(x)
    if update_i_return(n_layers): return x

    x=layers.Conv2D(512, (3,3), padding='same',activation=gdn.GDN(name="gdn_6"))(x)
    if update_i_return(n_layers): return x

    x=layers.Conv2D(1024, (3,3), padding='same',activation=gdn.GDN(name="gdn_7"))(x)
    if update_i_return(n_layers): return x


    x=layers.Flatten()(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(1024,activation='relu')(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(512,activation='relu')(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(256,activation='relu')(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(128,activation='relu')(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(64,activation='relu')(x)
    if update_i_return(n_layers): return x

    x=layers.Dense(latent_depth,activation='relu')(x)
    if update_i_return(n_layers): return x


    x_center_nu=layers.Dense(3*4,activation='sigmoid')(x[:,:3*4])
    #x_center_nu=tf.clip_by_value(x_center_nu, clip_value_min=0.0, clip_value_max=1)
    x_A_NN_diag=layers.Dense(2*4,activation='softplus')(x[:,3*4:5*4])
    x_A_NN_offdiag=layers.Dense(1*4)(x[:,5*4:6*4])
    x_A_NN_diag=tf.clip_by_value(x_A_NN_diag, clip_value_min=0.0, clip_value_max=50)
    x_A_NN_offdiag=tf.clip_by_value(x_A_NN_offdiag, clip_value_min=-50, clip_value_max=50)

    x = tf.concat([x_center_nu, x_A_NN_diag, x_A_NN_offdiag], axis=1)
    return x


def SSIMLoss(x, x_pred):
    ssim = tf.image.ssim(x, x_pred, max_val=1.0)
    ssim_mean = tf.reduce_mean(ssim)

    return 1 - ssim_mean

def MSSSIMLoss(x, x_pred):
    msssim = tf.image.ssim_multiscale(x, x_pred, max_val=1.0, filter_size=3, k2=0.2)
    msssim = tf.clip_by_value(msssim, 1e-8, 1.0)
    msssim_mean = tf.reduce_mean(msssim)
    #tf.debugging.check_numerics(msssim_mean, "NaN detected in MSSSIM computation")

    return 1 - msssim_mean



def Smoe_reconst_og(center, num_kernels, block_size):
    x = np.linspace(0, 1, block_size).astype(dtype=np.float32)
    y = np.linspace(0, 1, block_size).astype(dtype=np.float32)
    domain_init = np.array(np.meshgrid(x, y)).T
    domain_init = domain_init.reshape([block_size ** 2, 2])
    center_x = center[:, 0:num_kernels]
    center_y = center[:, num_kernels:2 * num_kernels]
    #A_NN = center[:, 3 * num_kernels:]
    #A_NN = np.reshape(A_NN, [-1, num_kernels, 2, 2])
    #A_NN = np.tril(A_NN)



    chol_diag = center[:, 3*num_kernels:5*num_kernels]
    chol_diag = np.reshape(chol_diag, (-1, num_kernels, 2))
    chol_offdiag = center[:, 5*num_kernels:]
    chol_offdiag = np.reshape(chol_offdiag, [-1, num_kernels, 1])
    zeros = np.zeros_like(chol_offdiag)
    upper = np.concatenate([chol_diag[:, :, :1], zeros], axis=-1)
    lower = np.concatenate([chol_offdiag, chol_diag[:, :, 1:]], axis=-1)
    chol = np.concatenate([upper, lower], axis=-1)
    chol = np.reshape(chol, (-1, num_kernels, 2, 2))
    A_NN = chol 
   
    nue_e=center[:,2*num_kernels:3*num_kernels]
    shape_x = np.shape(center_x)
    reshape_x = np.reshape(center_x, (shape_x[0], num_kernels, 1))
    reshape_y = np.reshape(center_y, (shape_x[0], num_kernels, 1))
    centers = np.reshape(tf.concat([reshape_x, reshape_y], axis=2), (shape_x[0], num_kernels, 2))
    nue_e=center[:,2*num_kernels:3*num_kernels]
    shape_x = np.shape(center_x)
    reshape_x = np.reshape(center_x, (shape_x[0], num_kernels, 1))
    reshape_y = np.reshape(center_y, (shape_x[0], num_kernels, 1))
    centers = np.reshape(tf.concat([reshape_x, reshape_y], axis=2), (shape_x[0], num_kernels, 2))

    musX = np.expand_dims(centers, axis=2)
    domain_exp = np.tile(np.expand_dims(np.expand_dims(domain_init, axis=0), axis=0),
                         (np.shape(musX)[0], np.shape(musX)[1], 1, 1))
    x_sub_mu = np.expand_dims(domain_exp.astype(dtype=np.float32) - musX.astype(dtype=np.float32),
                              axis=-1)
    n_exp = np.exp(
        np.negative(0.5 * np.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu, A_NN, A_NN, x_sub_mu)))

    n_w_norm = np.sum(n_exp, axis=1, keepdims=True)
    n_w_norm = np.maximum(10e-8, n_w_norm)

    w_e_op = np.divide(n_exp, n_w_norm)

    res = np.sum(w_e_op * np.expand_dims(nue_e.astype(dtype=np.float32), axis=-1), axis=1)
    res = np.minimum(np.maximum(res, 0), 1)
    res = np.reshape(res, (-1, block_size, block_size))
    # -----------------------------------------------------------------

    return res
