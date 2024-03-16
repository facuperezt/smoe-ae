#%%
import time
import gc 
import os
from os import walk
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras.layers import Input, Flatten, Lambda, Dense, Concatenate
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from skimage.metrics import structural_similarity as compare_ssim
import pickle
import matplotlib.pyplot as plt
from math import sqrt
#from keras import mixed_precision
#Scripts
from . import functions as fn
from . import classes as cl

from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

# Set global precision policy
# With this policy, layers use float16 computations and float32 variables
# mixed precision macht nur für gpu sinn, nicht für cpu, da noch langsamer als float32
# mixed_precision.set_global_policy('mixed_float16')
def load_model(n_layers):


    encoder_input = Input(shape=(BLOCK_SIZE,BLOCK_SIZE,1), name="encoder_input")
    model_no_Smoe = fn.standard_8(x=encoder_input, latent_depth=NUM_KERNELS*6, n_layers=n_layers)

    x=keras.Model(inputs=encoder_input,outputs=model_no_Smoe)
    g = []
    for (_, _, filenames) in walk(test_image_path):
        g.extend(filenames)
        break

    # for i in range(len(g)):
    #     filename = g[i]
    #     data = pickle.load(open(test_image_path + filename, "rb"))
    #     if STEP_VAL == BLOCK_SIZE:
    #         img_blocked = data["img_blocked"]
    #         img_blocked = np.reshape(img_blocked,(-1, img_blocked.shape[2], img_blocked.shape[3]))
    #         shape_orig = data["shape_orig"]
    #         shape_orig_blocked = data["shape_orig_blocked"]
    #         img_orig = np.reshape(img_blocked,(shape_orig_blocked[0], shape_orig_blocked[1], BLOCK_SIZE, BLOCK_SIZE)).transpose(
    #         0, 2, 1, 3).reshape(shape_orig[0], shape_orig[1])
    #         features = x.predict(img_blocked)
    #         print(features.shape)
    #         print(features.shape)
    #         break
    #%%
    try:
        y= cl.SmoeLayer(num_kernels=NUM_KERNELS, block_size=BLOCK_SIZE)(x.output)
        encoder_model=keras.Model(inputs=encoder_input,outputs=y)

        # Load the checkpoint
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")  
        encoder_model.load_weights(checkpoint_prefix.format(epoch=24))
        return encoder_model, g
    except ValueError:
        return x, g
#%%
BLOCK_SIZE = int(8)
STEP_VAL = int(8) #from blocked_images_bitstream_windows.py 
NUM_KERNELS = 4 #int(4*((BLOCK_SIZE)/16)*(BLOCK_SIZE)/16)
BATCH_SIZE = int(50)
EPOCHS = int(30)
BUFFER_SIZE = int(200)
User = "elmoghazi"
step_val = int(8)

#Path
checkpoint_dir = '../Trainingcheckpoint/' #Define the checkpoint directory to store the checkpoints after each epoch
test_image_path = '../pickles/' # Path test imgs used in Callback function
if __name__ == "__main__":
    encoder_model = load_model(100)
    d = {}
    for l in encoder_model.weights:
        print("_".join(l.name.split("/")[1:]))
        d.setdefault(l.name.split("/")[0], {})["_".join(l.name.split("/")[1:])] = l.numpy()
        
    import pickle
    print(d.keys())
    with open(f"tarek_tf_smoe_weights_and_biases_{BLOCK_SIZE}x{BLOCK_SIZE}_c.pkl", "wb") as f:
        pickle.dump(d, f)
    #%%

    g = []
    for (_, _, filenames) in walk(test_image_path):
        g.extend(filenames)
        break

    img_blocked = None
    for i in range(len(g)):
        filename = g[i]
        data = pickle.load(open(test_image_path + filename, "rb"))
        if STEP_VAL == BLOCK_SIZE:
            img_blocked = data["img_blocked"]
            img_blocked = np.reshape(img_blocked,(-1, img_blocked.shape[2], img_blocked.shape[3]))
            break
            shape_orig = data["shape_orig"]
            shape_orig_blocked = data["shape_orig_blocked"]
            img_orig = np.reshape(img_blocked,(shape_orig_blocked[0], shape_orig_blocked[1], BLOCK_SIZE, BLOCK_SIZE)).transpose(
            0, 2, 1, 3).reshape(shape_orig[0], shape_orig[1])
            features = x.predict(img_blocked)
            #img_reconst = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=NUM_KERNELS, block_size=BLOCK_SIZE, img_dim_blocked_x=shape_orig_blocked[0], img_dim_blocked_y=shape_orig_blocked[1], step_val=step_val)
            img_reconst = fn.Smoe_reconst_og(num_kernels=NUM_KERNELS,block_size=BLOCK_SIZE, center=features).reshape(shape_orig_blocked).transpose(0,2,1,3).reshape(shape_orig)
            psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
            ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)

            delta = np.abs(img_orig-img_reconst)

            fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
            ax1.tick_params(axis='both', labelsize=5)
            ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
            ax1.set_title('Original')
            ax2.tick_params(axis='both', labelsize=5)
            ax2.imshow(img_reconst, cmap='gray',
                    vmin=0, vmax=1)
            ax2.set_title(
            'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
            round(ssim_reconst, 2)))
            ax3.tick_params(axis='both', labelsize=5)
            im = ax3.imshow(delta, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im,ax=ax3)
            ax3.set_title('delta')
            plt.savefig('/home/elmoghazi/Masterarbeit/facundo/counter_'+str(i)+".png") # You need to change '/path/to/your/directory/your_image_name.png' to your actual path and file name
            plt.close(fig) # Close the figure to free up memory

            
        
        else:
            img_blocked = data["img_blocked"]
            #shape_orig = data["shape_orig"]
            #shape_orig_blocked = data["shape_orig_blocked"]
            #shape_step_val = data["shape_step_val"]
            #shape_step_val_blocked = data["shape_step_val_blocked"]
            ##overshoot_x = data["overshoot_x"]
            #overshoot_y = data["overshoot_y"]
            #img_orig = fn.average_overlap(img_blocked.reshape(shape_step_val_blocked[0], shape_step_val_blocked[1], BLOCK_SIZE, BLOCK_SIZE), block_size=BLOCK_SIZE, step_val=STEP_VAL, shape_orig=shape_orig, b_averag>
            #features = model_no_Smoe.predict(img_blocked)
            #img_reconst_overlap = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=NUM_KERNELS, block_size=BLOCK_SIZE, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=sha>
            #img_reconst = fn.Smoe_reconst_overlap(shape_orig=shape_orig, features=features, num_kernels=NUM_KERNELS, block_size=BLOCK_SIZE, img_dim_blocked_x=shape_step_val_blocked[0], img_dim_blocked_y=shape_step_>
            #psnr_reconst_overlap = fn.psnr(img_A=img_orig, img_B=img_reconst_overlap)
            #psnr_reconst = fn.psnr(img_A=img_orig, img_B=img_reconst)
            #ssim_reconst_overlap = compare_ssim(img_orig, img_reconst_overlap, data_range=1.)
            #ssim_reconst = compare_ssim(img_orig, img_reconst, data_range=1.)
        
            #fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
            #ax1.tick_params(axis='both', labelsize=5)
            #ax1.imshow(img_orig, cmap='gray', vmin=0, vmax=1)
            #ax1.set_title('Original')
            #ax2.tick_params(axis='both', labelsize=5)
            #ax2.imshow(img_reconst, cmap='gray',
            #        vmin=0, vmax=1)
            #ax2.set_title(
            #'psnr_reconst: ' + str(round(psnr_reconst, 2)) + ', ssim_reconst: ' + str(
            #round(ssim_reconst, 2)))

            #ax3.tick_params(axis='both', labelsize=5)
            #ax3.imshow(img_reconst_overlap, cmap='gray', vmin=0, vmax=1)
            #ax3.set_title(
            #                'psnr_smooth: ' + str(round(psnr_reconst_overlap, 2)) + ', ssim_smooth: ' + str(
            #round(ssim_reconst_overlap, 2)))
            #plt.show(block=False)

# %%
#%%

# layer1_real = [a.numpy() for a in x.weights if "layer_1" in a.name and "real" in a.name]
# layer1_imag = [a.numpy() for a in x.weights if "layer_1" in a.name and "imag" in a.name]
# layer1_bias = [a.numpy() for a in x.weights if "layer_1" in a.name and "bias" in a.name]
# layer1_gdn_beta = [a.numpy() for a in x.weights if "layer_1" in a.name and "beta" in a.name]
# layer1_gdn_gamma = [a.numpy() for a in x.weights if "layer_1" in a.name and "gamma" in a.name]

# layer2_real = [a.numpy() for a in x.weights if "layer_2" in a.name and "real" in a.name]
# layer2_imag = [a.numpy() for a in x.weights if "layer_2" in a.name and "imag" in a.name]
# layer2_bias = [a.numpy() for a in x.weights if "layer_2" in a.name and "bias" in a.name]
# layer2_gdn_beta = [a.numpy() for a in x.weights if "layer_2" in a.name and "beta" in a.name]
# layer2_gdn_gamma = [a.numpy() for a in x.weights if "layer_2" in a.name and "gamma" in a.name]

### PyTorch stuff ###
# import torch
# from gdn_pytorch import GDN

# import torch

# class RDFTConv(torch.nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, time_domain_size: Tuple[int, int] = (3, 3), freq_domain_size: Tuple[int, int] = (3, 2), corr: bool = True,
#                  stride: int = 2, padding: int = 1, device: torch.device = "cpu"):
#         """
#         I am fking guessing how this works in TF under the hood. But this in theory should work.
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.time_size = time_domain_size
#         self.freq_size = freq_domain_size
#         self.stride = stride
#         self.padding = padding
#         self.corr = corr
#         self.device = device
#         self.kernel_real = torch.nn.Parameter(torch.randn((out_channels, in_channels, *freq_domain_size)).to(device))
#         self.kernel_imag = torch.nn.Parameter(torch.randn((out_channels, in_channels, *freq_domain_size)).to(device))
#         self.bias = torch.nn.Parameter(torch.randn(out_channels).to(device))

#     def forward(self, x):
#         kernel = torch.fft.irfft2(torch.complex(self.kernel_real, self.kernel_imag).to(self.device), self.time_size)
#         print(kernel.shape)
#         if not self.corr:
#             kernel = kernel.flip(-2, -1)
#         x = torch.conv2d(x, kernel, self.bias, self.stride, padding=self.padding)
#         return x
    
# if len(layer1_imag) > 0 and len(layer1_real) > 0:
#     layer1_imag = layer1_imag[0]
#     layer1_real = layer1_real[0]
#     layer1_bias = layer1_bias[0]
#     layer1_gdn_beta = layer1_gdn_beta[0]
#     layer1_gdn_gamma = layer1_gdn_gamma[0]
#     corr_layer1_imag = torch.tensor(layer1_imag, dtype=torch.float32).cuda()
#     corr_layer1_real = torch.tensor(layer1_real, dtype=torch.float32).cuda()
#     corr_layer1_bias = torch.tensor(layer1_bias, dtype=torch.float32).cuda()

#     corr_layer1 = RDFTConv(1, 16, (3, 3), (3, 2), True, 2, 1, "cuda")
#     corr_layer1.kernel_imag.data = corr_layer1_imag.transpose(0, 1)
#     corr_layer1.kernel_real.data = corr_layer1_real.transpose(0, 1)
#     corr_layer1.bias.data = corr_layer1_bias

#     # corr_layer1_complex = torch.complex(corr_layer1_real, corr_layer1_imag).cuda()
#     # corr_layer1_kernel = torch.fft.irfft2(corr_layer1_complex, (3, 3)).permute(1, 0, 2, 3)
#     # corr_layer1_bias = torch.tensor(layer1_bias, dtype=torch.float32).cuda()

#     corr_layer1_gdn = GDN(16, "cuda")
#     corr_layer1_gdn.beta = torch.nn.Parameter(torch.tensor(layer1_gdn_beta, dtype=torch.float32, device="cuda").squeeze())
#     corr_layer1_gdn.gamma = torch.nn.Parameter(torch.tensor(layer1_gdn_gamma, dtype=torch.float32, device="cuda"))

# if len(layer2_imag) > 0 and len(layer2_real) > 0:
#     layer2_imag = layer2_imag[0]
#     layer2_real = layer2_real[0]
#     layer2_bias = layer2_bias[0]
#     layer2_gdn_beta = layer2_gdn_beta[0]
#     layer2_gdn_gamma = layer2_gdn_gamma[0]
#     corr_layer2_imag = torch.tensor(layer2_imag, dtype=torch.float32).cuda()
#     corr_layer2_real = torch.tensor(layer2_real, dtype=torch.float32).cuda()
#     corr_layer2_bias = torch.tensor(layer2_bias, dtype=torch.float32).cuda()

#     corr_layer2 = RDFTConv(16, 32, (3, 3), (3, 2), True, stride=2, padding=1, device="cuda")
#     corr_layer2.kernel_imag.data = corr_layer2_imag.transpose(0, 1)
#     corr_layer2.kernel_real.data = corr_layer2_real.transpose(0, 1)
#     corr_layer2.bias.data = corr_layer2_bias

#     # corr_layer2_complex = torch.complex(corr_layer2_real, corr_layer2_imag).cuda()
#     # corr_layer2_kernel = torch.fft.irfft2(corr_layer2_complex, (3, 3)).permute(1, 0, 2, 3)
#     # corr_layer2_bias = torch.tensor(layer2_bias, dtype=torch.float64).cuda()

#     corr_layer2_gdn = GDN(32, "cuda")
#     corr_layer2_gdn.beta = torch.nn.Parameter(torch.tensor(layer2_gdn_beta, dtype=torch.float32, device="cuda").squeeze())
#     corr_layer2_gdn.gamma = torch.nn.Parameter(torch.tensor(layer2_gdn_gamma, dtype=torch.float32, device="cuda"))

# out = torch.tensor(img_blocked, dtype=torch.float32).reshape(4096, 1, 8, 8).cuda()

# if len(layer1_imag) > 0:
#     out = corr_layer1(out)
#     print(out.shape)
#     out = corr_layer1_gdn(out)

# if len(layer2_imag) > 0:
#     out = corr_layer2(out)
#     print(out.shape)
#     out = corr_layer2_gdn(out)
