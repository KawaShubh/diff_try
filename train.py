# import yaml
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from models.vae import VAE
# from models.unet import WCESuperResUNet
# from utils.dataset import UnpairedWCEDataset
# from tqdm import tqdm
# import os
# from torchmetrics.image.lpip import LPIPS

# # Load config
# with open("configs/params.yaml") as f:
#     config = yaml.safe_load(f)

# # Initialize
# device = torch.device(config["device"])
# os.makedirs(config["paths"]["save_dir"], exist_ok=True)
# os.makedirs(config["paths"]["val_output_dir"], exist_ok=True)

# # Models
# vae = VAE(pretrained=config["model"]["use_pretrained_vae"]).to(device)
# unet = WCESuperResUNet().to(device)

# # Losses
# lpips = LPIPS(net_type='alex').to(device)  # Perceptual loss
# l1_loss = nn.L1Loss()

# # Optimizer
# optimizer = optim.AdamW(unet.parameters(), lr=config["train"]["lr"])

# # Datasets
# train_dataset = UnpairedWCEDataset(
#     hr_dir=config["data"]["hr_dir"],
#     lr_dir=config["data"]["lr_dir"]
# )
# val_dataset = UnpairedWCEDataset(
#     hr_dir=config["data"]["val_hr_dir"],
#     lr_dir=config["data"]["val_lr_dir"]
# )

# train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=1)  # Batch=1 for validation

# def validate_and_save(epoch):
#     unet.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):
#             lr, hr = batch["lr"].to(device), batch["hr"].to(device)
            
#             # Generate SR
#             lr_latent = vae.encode(lr)
#             pred_hr_latent = unet(lr_latent)
#             pred_hr = vae.decode(pred_hr_latent)
            
#             # Save images
#             if i < 3:  # Save first 3 samples
#                 save_image(pred_hr, f"{config['paths']['val_output_dir']}/epoch_{epoch}_sample_{i}.png")
#     unet.train()

# def save_image(tensor, path):
#     """Saves a tensor as PNG"""
#     img = (tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5
#     Image.fromarray(img.astype(np.uint8)).save(path)

# # Training loop
# for epoch in range(config["train"]["epochs"]):
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
#     for batch in progress_bar:
#         lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        
#         # Forward pass
#         hr_latent = vae.encode(hr)
#         lr_latent = vae.encode(lr)
#         pred_hr_latent = unet(lr_latent)
        
#         # Losses
#         l1 = l1_loss(pred_hr_latent, hr_latent)
#         pred_hr = vae.decode(pred_hr_latent)
#         perceptual = lpips(pred_hr, hr)  # Compare in image space
#         loss = config["train"]["loss_weights"]["l1"] * l1 + \
#                config["train"]["loss_weights"]["lpips"] * perceptual
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         progress_bar.set_postfix({"Loss": loss.item(), "L1": l1.item(), "LPIPS": perceptual.item()})
    
#     # Validation and checkpointing
#     if (epoch + 1) % config["train"]["val_interval"] == 0:
#         validate_and_save(epoch + 1)
    
#     if (epoch + 1) % config["train"]["save_interval"] == 0:
#         torch.save(unet.state_dict(), f"{config['paths']['save_dir']}/unet_epoch_{epoch+1}.pth")



import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vae import VAE
from models.unet import WCESuperResUNet
from utils.dataset import UnpairedWCEDataset
from tqdm import tqdm
import os
import torch.nn as nn
import lpips
# Load config
with open("configs/params.yaml") as f:
    config = yaml.safe_load(f)

# Initialize

import os


device = torch.device(config["train"]["device"])
print(device)
os.makedirs(config["paths"]["save_dir"], exist_ok=True)

# Models
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(pretrained=config["model"]["use_pretrained_vae"], device=device)

# vae = VAE(pretrained=).to(device)
unet = WCESuperResUNet().to(device)

# Losses
lpips_loss = lpips.LPIPS(net='alex').to(device)
l1_loss = nn.L1Loss()

# Optimizer
print(config["train"]["lr"])
optimizer = optim.AdamW(unet.parameters(), lr=float(config["train"]["lr"]))

# Datasets
train_dataset = UnpairedWCEDataset(
    hr_dir=config["data"]["hr_dir"],
    lr_dir=config["data"]["lr_dir"]
)

train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)

def save_image(tensor, path):
    """Saves a tensor as PNG"""
    img = (tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5
    Image.fromarray(img.astype(np.uint8)).save(path)

# Training loop
for epoch in range(config["train"]["epochs"]):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        lr, hr = batch["lr"].to(device), batch["hr"].to(device)
        
        # Forward pass
        hr_latent = vae.encode(hr)
        lr_latent = vae.encode(lr)
        pred_hr_latent = unet(lr_latent)
        
        # Losses
        print(f"Shape of pred_hr_latent: {pred_hr_latent.shape}")
        print(f"Shape of hr_latent: {hr_latent.shape}")

        l1 = l1_loss(pred_hr_latent, hr_latent)
        pred_hr = vae.decode(pred_hr_latent)
        perceptual = lpips_loss(pred_hr, hr)  # Compare in image space
        loss = config["train"]["loss_weights"]["l1"] * l1 + \
               config["train"]["loss_weights"]["lpips"] * perceptual
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Loss": loss.item(), "L1": l1.item(), "LPIPS": perceptual.item()})
    
    # Checkpointing
    if (epoch + 1) % config["train"]["save_interval"] == 0:
        torch.save(unet.state_dict(), f"{config['paths']['save_dir']}/unet_epoch_{epoch+1}.pth")
