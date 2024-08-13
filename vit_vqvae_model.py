#%%
import torch
import yaml
from transformers import ViTFeatureExtractor, ViTModel, ViTImageProcessor
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from datasets import load_dataset
dataset = load_dataset("food101", split="train")
from data.dataloader import DataLoader
#%%
train_loader = DataLoader(mode="dataset", data_path="professional_photos", img_size=384, block_size=16,
                            batch_size=1, n_kernels=4, kernels_outside=False, negative_experts=False,
                            device="cuda")
processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-384')
model = ViTModel.from_pretrained('google/vit-large-patch16-384', add_pooling_layer=False)
input_img = train_loader.get()
#%%
input_img = input_img.expand(input_img.shape[0], 3, input_img.shape[2], input_img.shape[3]).cuda().requires_grad_()
#%%
peft_cfg = LoraConfig(
    r=8,
    target_modules=["key", "query", "value"],
    lora_alpha=32,
    lora_dropout=0.05,
)
peft_model = get_peft_model(model, peft_cfg)

embedder = peft_model.cuda()
# del model

# %%
embedder = embedder.eval()
inputs = model(input_img)

# %%
%load_ext autoreload
%autoreload 2
from vq_vae.vqvae import VQVAE
import yaml
vqvae = VQVAE(
        in_channels=1024,
        num_hiddens=512,
        num_downsampling_layers=0,
        num_residual_layers=1,
        num_residual_hiddens=1,
        embedding_dim=28,
        num_embeddings=256,
        use_ema=False,
        decay=False,
        epsilon=1e-6,
        n_kernels=4,
        block_size=16,
        device="cuda"
        )
# %%
out = vqvae(inputs.permute(0,2,1))
# %%
